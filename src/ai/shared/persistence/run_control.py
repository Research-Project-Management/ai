"""Cooperative run cancellation registry (Redis or in-memory fallback)."""

import asyncio
import logging
import time

from redis.asyncio import Redis
from redis.exceptions import RedisError

from ai.shared.contracts.run_control import RunCancellationRegistry
from ai.shared.persistence.redis import await_redis

logger = logging.getLogger(__name__)


class InMemoryRunCancellationRegistry:
    """In-process cancellation flag store with TTL expiration."""

    def __init__(self, *, ttl_seconds: int = 300) -> None:
        self._ttl_seconds = ttl_seconds
        self._flags: dict[tuple[str, str], float] = {}
        self._lock = asyncio.Lock()

    async def clear(self, space_id: str, thread_id: str) -> None:
        async with self._lock:
            self._flags.pop((space_id, thread_id), None)

    async def request_cancel(self, space_id: str, thread_id: str) -> None:
        async with self._lock:
            self._flags[(space_id, thread_id)] = time.monotonic() + self._ttl_seconds

    async def is_cancelled(self, space_id: str, thread_id: str) -> bool:
        now = time.monotonic()
        async with self._lock:
            expires_at = self._flags.get((space_id, thread_id))
            if expires_at is None:
                return False
            if expires_at <= now:
                self._flags.pop((space_id, thread_id), None)
                return False
            return True


class RedisRunCancellationRegistry:
    """Distributed cooperative run cancellation backed by Redis keys."""

    def __init__(
        self,
        redis_client: Redis,
        *,
        ttl_seconds: int = 300,
        key_prefix: str = "ai:cancel",
    ) -> None:
        self._redis = redis_client
        self._ttl_seconds = ttl_seconds
        self._key_prefix = key_prefix

    def _key(self, space_id: str, thread_id: str) -> str:
        return f"{self._key_prefix}:{space_id}:{thread_id}"

    async def clear(self, space_id: str, thread_id: str) -> None:
        try:
            await await_redis(self._redis.delete(self._key(space_id, thread_id)))
        except (RedisError, OSError, TimeoutError) as exc:
            logger.warning("Redis clear cancel failed: %s", type(exc).__name__)

    async def request_cancel(self, space_id: str, thread_id: str) -> None:
        try:
            await await_redis(
                self._redis.set(self._key(space_id, thread_id), "1", ex=self._ttl_seconds)
            )
        except (RedisError, OSError, TimeoutError) as exc:
            logger.warning("Redis request cancel failed: %s", type(exc).__name__)

    async def is_cancelled(self, space_id: str, thread_id: str) -> bool:
        try:
            val = await await_redis(self._redis.get(self._key(space_id, thread_id)))
            return bool(val)
        except (RedisError, OSError, TimeoutError) as exc:
            logger.warning("Redis check cancel failed: %s", type(exc).__name__)
            return False


def build_run_cancellation(
    redis_client: Redis | None,
    *,
    ttl_seconds: int = 300,
) -> RunCancellationRegistry:
    """Build Redis or in-memory run cancellation registry."""
    if redis_client is None:
        return InMemoryRunCancellationRegistry(ttl_seconds=ttl_seconds)
    return RedisRunCancellationRegistry(redis_client, ttl_seconds=ttl_seconds)
