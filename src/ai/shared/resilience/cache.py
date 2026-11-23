"""Tool result TTL cache adapters (In-Memory & Redis) for idempotent read tools."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from ai.shared.persistence.redis import await_redis
from ai.shared.resilience.contracts import ToolCache

logger = logging.getLogger(__name__)


def build_cache_key(namespace: str, payload: dict[str, Any]) -> str:
    """Build a deterministic SHA-256 cache key from a namespace and argument dictionary."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    clean_ns = namespace.strip().strip(":")
    return f"{clean_ns}:{digest}" if clean_ns else digest


class InMemoryToolCache:
    """Async-safe in-memory TTL cache with bounded capacity for local dev & tests."""

    def __init__(self, *, max_entries: int = 512) -> None:
        self._max_entries = max(1, max_entries)
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        now = time.monotonic()
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if now >= expires_at:
                del self._store[key]
                return None
            return value

    async def set(self, key: str, value: Any, *, ttl_seconds: int = 300) -> None:
        if ttl_seconds <= 0:
            return
        expires_at = time.monotonic() + ttl_seconds
        async with self._lock:
            now = time.monotonic()
            expired_keys = [k for k, (exp, _) in self._store.items() if now >= exp]
            for k in expired_keys:
                del self._store[k]

            self._store.pop(key, None)
            while len(self._store) >= self._max_entries:
                oldest_key = next(iter(self._store))
                del self._store[oldest_key]
            self._store[key] = (expires_at, value)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)


class RedisToolCache:
    """Fail-open Redis TTL cache adapter storing JSON-serialized tool outputs."""

    def __init__(
        self,
        redis: Redis,
        *,
        key_prefix: str = "flux:toolcache:",
    ) -> None:
        self._redis = redis
        self._prefix = key_prefix

    def _prefixed(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def get(self, key: str) -> Any | None:
        try:
            raw = await await_redis(self._redis.get(self._prefixed(key)))
            if raw is None:
                return None
            return json.loads(raw)
        except (RedisError, TypeError, ValueError) as exc:
            logger.warning("RedisToolCache.get failed (fail-open) for %s: %s", key, exc)
            return None

    async def set(self, key: str, value: Any, *, ttl_seconds: int = 300) -> None:
        if ttl_seconds <= 0:
            return
        try:
            serialized = json.dumps(value, default=str)
            await await_redis(self._redis.set(self._prefixed(key), serialized, ex=ttl_seconds))
        except (RedisError, TypeError, ValueError) as exc:
            logger.warning("RedisToolCache.set failed (fail-open) for %s: %s", key, exc)

    async def delete(self, key: str) -> None:
        try:
            await await_redis(self._redis.delete(self._prefixed(key)))
        except RedisError as exc:
            logger.warning("RedisToolCache.delete failed (fail-open) for %s: %s", key, exc)


def create_tool_cache(
    redis: Redis | None,
    *,
    key_prefix: str = "flux:toolcache:",
    max_memory_entries: int = 512,
) -> ToolCache:
    """Instantiate RedisToolCache when Redis is configured, falling back to InMemoryToolCache."""
    if redis is not None:
        return RedisToolCache(redis, key_prefix=key_prefix)
    return InMemoryToolCache(max_entries=max_memory_entries)
