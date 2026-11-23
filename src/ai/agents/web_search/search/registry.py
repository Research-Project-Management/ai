"""Pluggable SearchProviderRegistry with concurrent fan-out and fault isolation."""

import asyncio
import logging
from collections.abc import Sequence

import httpx

from ai.agents.web_search.search.contracts import (
    SearchCache,
    SearchCategoryFilter,
    SearchHit,
    SearchProvider,
    SearchQuery,
)
from ai.agents.web_search.search.providers import (
    CachedSearchProvider,
    FallbackSearchProvider,
    SearXNGSearchProvider,
    SerperGoogleSearchProvider,
    TavilySearchProvider,
)
from ai.agents.web_search.search.scholar import (
    ArxivSearchProvider,
    CrossrefSearchProvider,
    OpenAlexSearchProvider,
)
from ai.shared.config.web_search import WebSearchConfig


class SearchProviderRegistry:
    """Runtime registry supporting hot-swap, addition, removal, and parallel fan-out."""

    def __init__(self, providers: Sequence[SearchProvider] = ()) -> None:
        self._providers: dict[str, SearchProvider] = {}
        for provider in providers:
            self.register(provider, replace=False)

    def register(self, provider: SearchProvider, *, replace: bool = True) -> None:
        if not replace and provider.name in self._providers:
            raise ValueError(f"Duplicate search provider name: '{provider.name}'")
        self._providers[provider.name] = provider

    def get(self, name: str) -> SearchProvider | None:
        return self._providers.get(name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._providers.keys())

    def by_category(self, category: SearchCategoryFilter) -> tuple[SearchProvider, ...]:
        if category == "hybrid":
            return tuple(self._providers.values())
        return tuple(p for p in self._providers.values() if p.category == category)

    async def search_many(
        self,
        queries: Sequence[SearchQuery],
        *,
        timeout_seconds: float = 10.0,
    ) -> tuple[SearchHit, ...]:
        """Fan out all matching `(query, provider)` pairs concurrently with graceful failure."""
        coros = []
        for query in queries:
            providers = self.by_category(query.category)
            if not providers and query.category != "hybrid":
                # Graceful fallback to any available providers when requested category has none
                providers = self.by_category("hybrid")
            for provider in providers:
                coros.append(
                    self._run_provider_safe(provider, query, timeout_seconds=timeout_seconds)
                )

        if not coros:
            return ()

        batches = await asyncio.gather(*coros)
        merged: list[SearchHit] = []
        for batch in batches:
            merged.extend(batch)
        return tuple(merged)

    @staticmethod
    async def _run_provider_safe(
        provider: SearchProvider,
        query: SearchQuery,
        *,
        timeout_seconds: float,
    ) -> tuple[SearchHit, ...]:
        try:
            return await asyncio.wait_for(provider.search(query), timeout=timeout_seconds)
        except (httpx.HTTPError, TimeoutError, ValueError, RuntimeError, OSError) as exc:
            logging.getLogger(__name__).warning(
                "Provider '%s' degraded during parallel search: %s: %s",
                provider.name,
                type(exc).__name__,
                exc,
            )
            return ()


def _wrap_cache(
    provider: SearchProvider,
    cache: SearchCache | None,
    *,
    ttl_seconds: int,
) -> SearchProvider:
    if cache is None or ttl_seconds <= 0:
        return provider
    return CachedSearchProvider(provider, cache, ttl_seconds=ttl_seconds)


def build_default_search_registry(
    config: WebSearchConfig,
    *,
    cache: SearchCache | None = None,
    allow_remote: bool = True,
    http_client: httpx.AsyncClient | None = None,
) -> SearchProviderRegistry:
    """Build default web + scholarly registry from `WebSearchConfig`."""
    registry = SearchProviderRegistry()
    if not allow_remote:
        return registry

    web_chain: list[SearchProvider] = []
    if config.has_searxng:
        web_chain.append(
            SearXNGSearchProvider(
                config.searxng_url,
                timeout_seconds=config.timeout_seconds,
                user_agent=config.user_agent,
                http_client=http_client,
            )
        )
    if config.has_tavily:
        web_chain.append(
            TavilySearchProvider(
                config.tavily_api_key.get_secret_value(),
                base_url=config.tavily_base_url,
                timeout_seconds=config.timeout_seconds,
                http_client=http_client,
            )
        )
    if config.has_serper:
        web_chain.append(
            SerperGoogleSearchProvider(
                config.serper_api_key.get_secret_value(),
                base_url=config.serper_base_url,
                timeout_seconds=config.timeout_seconds,
                http_client=http_client,
            )
        )

    if len(web_chain) == 1:
        registry.register(_wrap_cache(web_chain[0], cache, ttl_seconds=config.cache_ttl_seconds))
    elif len(web_chain) > 1:
        composite = FallbackSearchProvider(name="web_default", category="web", providers=web_chain)
        registry.register(_wrap_cache(composite, cache, ttl_seconds=config.cache_ttl_seconds))

    scholar_providers: tuple[SearchProvider, ...] = (
        OpenAlexSearchProvider(
            base_url=config.openalex_base_url,
            api_key=config.openalex_api_key.get_secret_value(),
            timeout_seconds=config.timeout_seconds,
            user_agent=config.user_agent,
            http_client=http_client,
        ),
        ArxivSearchProvider(
            base_url=config.arxiv_base_url,
            timeout_seconds=config.timeout_seconds,
            user_agent=config.user_agent,
            http_client=http_client,
        ),
        CrossrefSearchProvider(
            base_url=config.crossref_base_url,
            timeout_seconds=config.timeout_seconds,
            user_agent=config.user_agent,
            http_client=http_client,
        ),
    )
    for provider in scholar_providers:
        registry.register(_wrap_cache(provider, cache, ttl_seconds=config.cache_ttl_seconds))

    return registry
