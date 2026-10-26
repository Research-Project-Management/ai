"""Search/fetch cache: a typed facade over the shared `ToolCache` (Redis or in-memory)."""

import hashlib
import json
from typing import Any

from redis.asyncio import Redis

from ai.agents.web_search.search.contracts import SearchCache, SearchHit, SearchQuery
from ai.shared.resilience.cache import create_tool_cache
from ai.shared.resilience.contracts import ToolCache


def make_search_cache_key(provider_name: str, query: SearchQuery) -> str:
    """Build a deterministic SHA-256 cache key for a provider query."""
    payload = {
        "provider": provider_name,
        "query": query.query.lower(),
        "category": query.category,
        "max_results": query.max_results,
        "allowed_domains": list(query.allowed_domains),
        "blocked_domains": list(query.blocked_domains),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"ai:ws:search:{provider_name}:{digest}"


class NoOpSearchCache:
    """Zero-op cache implementation when caching is disabled (`ttl_seconds=0`)."""

    async def get_hits(self, key: str) -> tuple[SearchHit, ...] | None:
        del key
        return None

    async def set_hits(self, key: str, hits: tuple[SearchHit, ...], *, ttl_seconds: int) -> None:
        del key, hits, ttl_seconds

    async def get_json(self, key: str) -> dict[str, Any] | None:
        del key
        return None

    async def set_json(self, key: str, payload: dict[str, Any], *, ttl_seconds: int) -> None:
        del key, payload, ttl_seconds


class ToolCacheSearchCache:
    """Stores search hits and fetch payloads as JSON values in a shared `ToolCache`."""

    def __init__(self, backend: ToolCache) -> None:
        self._backend = backend

    async def get_hits(self, key: str) -> tuple[SearchHit, ...] | None:
        raw = await self._backend.get(key)
        if not isinstance(raw, list):
            return None
        return tuple(SearchHit.model_validate(item) for item in raw if isinstance(item, dict))

    async def set_hits(self, key: str, hits: tuple[SearchHit, ...], *, ttl_seconds: int) -> None:
        if hits:
            await self._backend.set(
                key, [hit.model_dump(mode="json") for hit in hits], ttl_seconds=ttl_seconds
            )

    async def get_json(self, key: str) -> dict[str, Any] | None:
        raw = await self._backend.get(key)
        return raw if isinstance(raw, dict) else None

    async def set_json(self, key: str, payload: dict[str, Any], *, ttl_seconds: int) -> None:
        await self._backend.set(key, payload, ttl_seconds=ttl_seconds)


def build_search_cache(redis_client: Redis | None, *, ttl_seconds: int) -> SearchCache:
    """Return a Redis-backed cache when configured, in-memory when TTL > 0, else no-op."""
    if ttl_seconds <= 0:
        return NoOpSearchCache()
    return ToolCacheSearchCache(create_tool_cache(redis_client))
