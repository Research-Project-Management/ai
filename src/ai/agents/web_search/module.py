"""Assembled `web_search_agent` vertical-slice module."""

from dataclasses import dataclass

import httpx
from redis.asyncio import Redis

from ai.agents.web_search.fetch.ladder import UrlFetchLadder, build_default_fetch_ladder
from ai.agents.web_search.fetch.security import DnsResolver
from ai.agents.web_search.search.cache import build_search_cache
from ai.agents.web_search.search.contracts import SearchCache
from ai.agents.web_search.search.registry import (
    SearchProviderRegistry,
    build_default_search_registry,
)
from ai.agents.web_search.workflow.contracts import WebSearchAgentConfig
from ai.agents.web_search.workflow.graph import build_web_search_agent_spec
from ai.orchestrator.contracts import AgentSpec
from ai.shared.config.web_search import WebSearchConfig
from ai.shared.llm.factory import ChatRunnable


@dataclass(frozen=True)
class WebSearchModule:
    """Assembled web search & fetch dependencies exposed for Composition Root wiring."""

    search_cache: SearchCache
    search_registry: SearchProviderRegistry
    fetch_ladder: UrlFetchLadder
    agent_spec: AgentSpec


def build_web_search_module(
    *,
    llm: ChatRunnable,
    config: WebSearchConfig | None = None,
    agent_config: WebSearchAgentConfig | None = None,
    redis_client: Redis | None = None,
    search_cache: SearchCache | None = None,
    search_registry: SearchProviderRegistry | None = None,
    fetch_ladder: UrlFetchLadder | None = None,
    dns_resolver: DnsResolver | None = None,
    http_client: httpx.AsyncClient | None = None,
    allow_remote: bool = True,
) -> WebSearchModule:
    """Assemble pluggable search registry, fetch escalation ladder, and `web_search_agent`."""
    ws_cfg = config or WebSearchConfig()
    active_cache = search_cache or build_search_cache(
        redis_client,
        ttl_seconds=ws_cfg.cache_ttl_seconds,
    )
    active_registry = search_registry or build_default_search_registry(
        ws_cfg,
        cache=active_cache,
        allow_remote=allow_remote,
        http_client=http_client,
    )
    active_ladder = fetch_ladder or build_default_fetch_ladder(
        ws_cfg,
        cache=active_cache,
        allow_remote=allow_remote,
        dns_resolver=dns_resolver,
        http_client=http_client,
    )
    eff_agent_cfg = agent_config or WebSearchAgentConfig(
        top_k_evidence=ws_cfg.top_k_evidence,
        max_deep_fetch_urls=ws_cfg.max_deep_fetch_urls,
        max_results_per_provider=ws_cfg.max_results_per_provider,
        min_relevance_score=ws_cfg.min_relevance_score,
        search_timeout_seconds=ws_cfg.timeout_seconds,
    )
    spec = build_web_search_agent_spec(
        llm,
        active_registry,
        active_ladder,
        config=eff_agent_cfg,
    )
    return WebSearchModule(
        search_cache=active_cache,
        search_registry=active_registry,
        fetch_ladder=active_ladder,
        agent_spec=spec,
    )


__all__ = ["WebSearchModule", "build_web_search_module"]
