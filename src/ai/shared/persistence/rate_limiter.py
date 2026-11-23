"""Sliding-window rate limiter adapters scoped per Flux `space_id`."""

import asyncio
import logging
import math
import time
from collections import defaultdict, deque
from uuid import uuid4

from redis.asyncio import Redis
from redis.exceptions import RedisError

from ai.shared.config.rate_limit import RateLimitConfig
from ai.shared.contracts.rate_limit import RateLimiter, RateLimitResult
from ai.shared.persistence.redis import await_redis

logger = logging.getLogger(__name__)

_SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]
local cutoff = now - window

redis.call("ZREMRANGEBYSCORE", key, "-inf", cutoff)
local current = redis.call("ZCARD", key)

if current < limit then
    redis.call("ZADD", key, now, member)
    redis.call("PEXPIRE", key, window)
    return {1, limit - current - 1, math.ceil(window / 1000)}
end

local oldest = redis.call("ZRANGE", key, 0, 0, "WITHSCORES")
local reset_sec = math.ceil(window / 1000)
if oldest[2] then
    local elapsed = now - tonumber(oldest[2])
    local remaining_ms = math.max(0, window - elapsed)
    reset_sec = math.max(1, math.ceil(remaining_ms / 1000))
end
redis.call("PEXPIRE", key, window)
return {0, 0, reset_sec}
"""


class NoOpRateLimiter:
    """Rate limiter used when `RATE_LIMIT_ENABLED=false`."""

    def __init__(self, *, limit: int = 0) -> None:
        self._limit = max(0, limit)

    async def check(self, key: str) -> RateLimitResult:
        del key
        return RateLimitResult(
            allowed=True,
            limit=self._limit,
            remaining=self._limit,
            reset_after_seconds=0,
            reason="disabled",
        )


class InMemorySlidingWindowRateLimiter:
    """Local sliding-window fallback used when Redis URL is empty in dev/tests."""

    def __init__(
        self,
        *,
        limit: int,
        window_seconds: int,
        max_buckets: int = 4096,
    ) -> None:
        self._limit = limit
        self._window_seconds = window_seconds
        self._max_buckets = max(64, max_buckets)
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> RateLimitResult:
        now = time.monotonic()
        cutoff = now - self._window_seconds
        async with self._lock:
            if len(self._buckets) >= self._max_buckets and key not in self._buckets:
                stale = [k for k, b in self._buckets.items() if not b or b[-1] <= cutoff]
                for k in stale:
                    del self._buckets[k]
            bucket = self._buckets[key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            if len(bucket) < self._limit:
                bucket.append(now)
                remaining = self._limit - len(bucket)
                return RateLimitResult(
                    allowed=True,
                    limit=self._limit,
                    remaining=remaining,
                    reset_after_seconds=self._window_seconds,
                )

            reset_sec = max(1, math.ceil(self._window_seconds - (now - bucket[0])))
            return RateLimitResult(
                allowed=False,
                limit=self._limit,
                remaining=0,
                reset_after_seconds=reset_sec,
                reason="rate_limit_exceeded",
            )


class RedisSlidingWindowRateLimiter:
    """Distributed atomic sliding-window rate limiter backed by Redis ZSET + Lua."""

    def __init__(
        self,
        redis_client: Redis,
        *,
        limit: int,
        window_seconds: int,
        fail_open: bool = True,
        key_prefix: str = "ai:rl:space",
    ) -> None:
        self._redis = redis_client
        self._limit = limit
        self._window_seconds = window_seconds
        self._fail_open = fail_open
        self._key_prefix = key_prefix

    async def check(self, key: str) -> RateLimitResult:
        redis_key = f"{self._key_prefix}:{key}"
        now_ms = int(time.time() * 1000)
        window_ms = self._window_seconds * 1000
        member = f"{now_ms}:{uuid4().hex[:8]}"

        try:
            raw = await await_redis(
                self._redis.eval(
                    _SLIDING_WINDOW_LUA,
                    1,
                    redis_key,
                    str(now_ms),
                    str(window_ms),
                    str(self._limit),
                    member,
                )
            )
            allowed_flag, remaining, reset_sec = int(raw[0]), int(raw[1]), int(raw[2])
            allowed = allowed_flag == 1
            return RateLimitResult(
                allowed=allowed,
                limit=self._limit,
                remaining=max(0, remaining),
                reset_after_seconds=max(0, reset_sec),
                reason="" if allowed else "rate_limit_exceeded",
            )
        except (RedisError, OSError, TimeoutError) as exc:
            logger.warning(
                "Redis rate limiter error for space=%s (fail_open=%s): %s",
                key,
                self._fail_open,
                type(exc).__name__,
            )
            if self._fail_open:
                return RateLimitResult(
                    allowed=True,
                    limit=self._limit,
                    remaining=self._limit,
                    reset_after_seconds=self._window_seconds,
                    reason="redis_unavailable_fail_open",
                )
            return RateLimitResult(
                allowed=False,
                limit=self._limit,
                remaining=0,
                reset_after_seconds=self._window_seconds,
                reason="redis_unavailable_fail_closed",
            )


def build_rate_limiter(
    config: RateLimitConfig,
    redis_client: Redis | None,
) -> RateLimiter:
    """Construct the configured per-space RateLimiter strategy."""
    if not config.enabled:
        return NoOpRateLimiter(limit=config.requests_per_window)
    if redis_client is None:
        return InMemorySlidingWindowRateLimiter(
            limit=config.requests_per_window,
            window_seconds=config.window_seconds,
        )
    return RedisSlidingWindowRateLimiter(
        redis_client=redis_client,
        limit=config.requests_per_window,
        window_seconds=config.window_seconds,
        fail_open=config.fail_open,
    )
