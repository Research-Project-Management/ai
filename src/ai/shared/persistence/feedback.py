"""Structured feedback ingestion adapters (In-Memory & Redis) for run quality evaluation."""

import asyncio
import json
import logging
from collections import deque

from redis.asyncio import Redis
from redis.exceptions import RedisError

from ai.shared.contracts.run_control import FeedbackRecord, FeedbackSink
from ai.shared.persistence.redis import await_redis

logger = logging.getLogger(__name__)


class InMemoryFeedbackSink:
    """Bounded thread-safe feedback recorder with space-scoped lookup."""

    def __init__(self, *, max_items: int = 2000) -> None:
        self._items: deque[FeedbackRecord] = deque(maxlen=max_items)
        self._lock = asyncio.Lock()

    async def record(self, item: FeedbackRecord) -> None:
        async with self._lock:
            self._items.append(item)
        logger.info(
            "run.feedback trace_id=%s space_id=%s score=%s category=%s",
            item.trace_id,
            item.space_id,
            item.score,
            item.category,
        )

    async def list_by_trace(self, trace_id: str, *, space_id: str) -> list[FeedbackRecord]:
        async with self._lock:
            return [
                item
                for item in self._items
                if item.trace_id == trace_id and item.space_id == space_id
            ]


class RedisFeedbackSink:
    """Fail-open persistent Redis feedback adapter with space isolation and in-memory fallback."""

    def __init__(
        self,
        redis: Redis,
        *,
        key_prefix: str = "flux:feedback:",
        ttl_seconds: int = 7 * 86_400,
    ) -> None:
        self._redis = redis
        self._prefix = key_prefix
        self._ttl_seconds = max(60, ttl_seconds)
        self._fallback = InMemoryFeedbackSink()

    def _key(self, space_id: str, trace_id: str) -> str:
        return f"{self._prefix}{space_id}:{trace_id}"

    async def record(self, item: FeedbackRecord) -> None:
        await self._fallback.record(item)
        key = self._key(item.space_id, item.trace_id)
        try:
            payload = json.dumps(item.model_dump(mode="json"))
            await await_redis(self._redis.rpush(key, payload))
            await await_redis(self._redis.expire(key, self._ttl_seconds))
        except (RedisError, OSError, TimeoutError, TypeError, ValueError) as exc:
            logger.warning("RedisFeedbackSink.record failed (fail-open): %s", exc)

    async def list_by_trace(self, trace_id: str, *, space_id: str) -> list[FeedbackRecord]:
        key = self._key(space_id, trace_id)
        try:
            raw_items = await await_redis(self._redis.lrange(key, 0, 99))
            if raw_items:
                return [FeedbackRecord.model_validate(json.loads(raw)) for raw in raw_items]
        except (RedisError, OSError, TimeoutError, TypeError, ValueError) as exc:
            logger.warning("RedisFeedbackSink.list_by_trace failed (fail-open): %s", exc)
        return await self._fallback.list_by_trace(trace_id, space_id=space_id)


def build_feedback_sink(redis_client: Redis | None) -> FeedbackSink:
    """Create `RedisFeedbackSink` when Redis is configured, else `InMemoryFeedbackSink`."""
    if redis_client is not None:
        return RedisFeedbackSink(redis_client)
    return InMemoryFeedbackSink()
