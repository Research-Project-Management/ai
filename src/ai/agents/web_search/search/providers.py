"""Web `SearchProvider` adapters (SearXNG, Tavily, Serper), failover chain, and cache decorator."""

import logging
from collections.abc import Sequence
from typing import Any

import httpx

from ai.agents.web_search.search.cache import make_search_cache_key
from ai.agents.web_search.search.contracts import (
    ProviderCategory,
    SearchCache,
    SearchHit,
    SearchProvider,
    SearchQuery,
)
from ai.agents.web_search.search.http import http_request
from ai.agents.web_search.search.scope import apply_site_operators, url_matches_domain_scope

_LOGGER = logging.getLogger(__name__)


class SearXNGSearchProvider:
    """Self-hosted SearXNG JSON metasearch adapter (`GET /search?format=json`)."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 10.0,
        user_agent: str = "FluxResearchAI/0.1",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.strip().rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._user_agent = user_agent
        self._http_client = http_client

    @property
    def name(self) -> str:
        return "searxng"

    @property
    def category(self) -> ProviderCategory:
        return "web"

    async def search(self, query: SearchQuery) -> tuple[SearchHit, ...]:
        scoped_query = apply_site_operators(
            query.query,
            allowed_domains=query.allowed_domains,
            blocked_domains=query.blocked_domains,
        )
        response = await http_request(
            method="GET",
            url=f"{self._base_url}/search",
            http_client=self._http_client,
            timeout_seconds=self._timeout_seconds,
            headers={"User-Agent": self._user_agent, "Accept": "application/json"},
            params={"q": scoped_query, "format": "json"},
        )
        response.raise_for_status()
        payload = response.json()
        raw_results = payload.get("results") if isinstance(payload, dict) else []
        if not isinstance(raw_results, list):
            return ()

        hits: list[SearchHit] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            title = str(item.get("title") or url).strip()
            if not url or not title:
                continue
            if not url_matches_domain_scope(
                url,
                allowed_domains=query.allowed_domains,
                blocked_domains=query.blocked_domains,
            ):
                continue
            snippet = str(item.get("content") or "").strip()
            raw_score = item.get("score")
            score = max(0.0, float(raw_score)) if isinstance(raw_score, int | float) else 0.0
            hits.append(
                SearchHit(
                    provider=self.name,
                    category=self.category,
                    title=title,
                    url=url,
                    snippet=snippet,
                    score=score,
                )
            )
            if len(hits) >= query.max_results:
                break
        return tuple(hits)


class TavilySearchProvider:
    """Tavily AI Search REST API adapter supporting domain allow/deny lists and raw content."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.tavily.com",
        timeout_seconds: float = 10.0,
        include_raw_content: bool = True,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key.strip()
        self._base_url = base_url.strip().rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._include_raw_content = include_raw_content
        self._http_client = http_client

    @property
    def name(self) -> str:
        return "tavily"

    @property
    def category(self) -> ProviderCategory:
        return "web"

    async def search(self, query: SearchQuery) -> tuple[SearchHit, ...]:
        if not self._api_key:
            return ()
        body: dict[str, Any] = {
            "api_key": self._api_key,
            "query": query.query,
            "max_results": query.max_results,
            "include_raw_content": self._include_raw_content,
        }
        if query.allowed_domains:
            body["include_domains"] = list(query.allowed_domains)
        elif query.blocked_domains:
            body["exclude_domains"] = list(query.blocked_domains)

        response = await http_request(
            method="POST",
            url=f"{self._base_url}/search",
            http_client=self._http_client,
            timeout_seconds=self._timeout_seconds,
            headers={"Content-Type": "application/json"},
            json_body=body,
        )
        response.raise_for_status()
        payload = response.json()
        raw_results = payload.get("results") if isinstance(payload, dict) else []
        if not isinstance(raw_results, list):
            return ()

        hits: list[SearchHit] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            title = str(item.get("title") or url).strip()
            if not url or not title:
                continue
            if not url_matches_domain_scope(
                url,
                allowed_domains=query.allowed_domains,
                blocked_domains=query.blocked_domains,
            ):
                continue
            snippet = str(item.get("content") or "").strip()
            raw_content = item.get("raw_content")
            content = raw_content.strip() if isinstance(raw_content, str) else ""
            raw_score = item.get("score")
            score = max(0.0, float(raw_score)) if isinstance(raw_score, int | float) else 0.0
            hits.append(
                SearchHit(
                    provider=self.name,
                    category=self.category,
                    title=title,
                    url=url,
                    snippet=snippet,
                    content=content,
                    score=score,
                )
            )
            if len(hits) >= query.max_results:
                break
        return tuple(hits)


class SerperGoogleSearchProvider:
    """Serper Google SERP JSON adapter (`POST /search`)."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://google.serper.dev",
        timeout_seconds: float = 10.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key.strip()
        self._base_url = base_url.strip().rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._http_client = http_client

    @property
    def name(self) -> str:
        return "serper_google"

    @property
    def category(self) -> ProviderCategory:
        return "web"

    async def search(self, query: SearchQuery) -> tuple[SearchHit, ...]:
        if not self._api_key:
            return ()
        scoped_query = apply_site_operators(
            query.query,
            allowed_domains=query.allowed_domains,
            blocked_domains=query.blocked_domains,
        )
        response = await http_request(
            method="POST",
            url=f"{self._base_url}/search",
            http_client=self._http_client,
            timeout_seconds=self._timeout_seconds,
            headers={"X-API-KEY": self._api_key, "Content-Type": "application/json"},
            json_body={"q": scoped_query, "num": query.max_results},
        )
        response.raise_for_status()
        payload = response.json()
        organic = payload.get("organic") if isinstance(payload, dict) else []
        if not isinstance(organic, list):
            return ()

        hits: list[SearchHit] = []
        for idx, item in enumerate(organic):
            if not isinstance(item, dict):
                continue
            url = str(item.get("link") or "").strip()
            title = str(item.get("title") or url).strip()
            if not url or not title:
                continue
            if not url_matches_domain_scope(
                url,
                allowed_domains=query.allowed_domains,
                blocked_domains=query.blocked_domains,
            ):
                continue
            snippet = str(item.get("snippet") or "").strip()
            hits.append(
                SearchHit(
                    provider=self.name,
                    category=self.category,
                    title=title,
                    url=url,
                    snippet=snippet,
                    score=max(0.1, 1.0 - idx * 0.08),
                )
            )
            if len(hits) >= query.max_results:
                break
        return tuple(hits)


class FallbackSearchProvider:
    """Chain-of-Responsibility composite returning hits from the first healthy provider."""

    def __init__(
        self,
        *,
        name: str,
        category: ProviderCategory,
        providers: Sequence[SearchProvider],
    ) -> None:
        self._name = name
        self._category: ProviderCategory = category
        self._providers = tuple(providers)

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> ProviderCategory:
        return self._category

    async def search(self, query: SearchQuery) -> tuple[SearchHit, ...]:
        for provider in self._providers:
            try:
                hits = await provider.search(query)
            except (httpx.HTTPError, TimeoutError, ValueError, RuntimeError, OSError) as exc:
                _LOGGER.warning(
                    "Search provider '%s' failed cleanly: %s: %s",
                    provider.name,
                    type(exc).__name__,
                    exc,
                )
                continue
            if hits:
                return hits
        return ()


class CachedSearchProvider:
    """Decorator caching non-empty results from an underlying `SearchProvider`."""

    def __init__(
        self,
        inner: SearchProvider,
        cache: SearchCache,
        *,
        ttl_seconds: int = 1800,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._ttl_seconds = ttl_seconds

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def category(self) -> ProviderCategory:
        return self._inner.category

    async def search(self, query: SearchQuery) -> tuple[SearchHit, ...]:
        key = make_search_cache_key(self._inner.name, query)
        cached = await self._cache.get_hits(key)
        if cached is not None:
            return cached
        hits = await self._inner.search(query)
        if hits:
            await self._cache.set_hits(key, hits, ttl_seconds=self._ttl_seconds)
        return hits
