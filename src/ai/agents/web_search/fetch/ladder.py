"""Chain-of-Responsibility URL Fetch Escalation Ladder (Client IP -> Open Mirror -> Safe HTTP)."""

import asyncio
import hashlib
import json
from collections.abc import Sequence

import httpx

from ai.agents.web_search.fetch.client_stage import ClientSuppliedFetchStage
from ai.agents.web_search.fetch.contracts import (
    TERMINAL_SECURITY_BLOCKS,
    AcademicIdentifiers,
    ClientSourcePayload,
    FetchRequest,
    FetchResult,
    UrlFetchStage,
)
from ai.agents.web_search.fetch.extractor import (
    extract_academic_identifiers,
)
from ai.agents.web_search.fetch.http_stage import SafeHttpFetchStage
from ai.agents.web_search.fetch.mirror_stage import OpenAcademicMirrorStage
from ai.agents.web_search.fetch.security import (
    DnsResolver,
)
from ai.agents.web_search.ranking.deduplicator import canonicalize_url
from ai.agents.web_search.search.contracts import SearchCache
from ai.shared.config.web_search import WebSearchConfig


def _fetch_cache_key(request: FetchRequest) -> str:
    payload = {
        "url": canonicalize_url(request.url),
        "allowed": list(request.allowed_domains),
        "blocked": list(request.blocked_domains),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"ai:ws:fetch:{digest}"


class UrlFetchLadder:
    """Orchestrates pluggable `UrlFetchStage` adapters in escalating order."""

    def __init__(
        self,
        stages: Sequence[UrlFetchStage] = (),
        *,
        cache: SearchCache | None = None,
        cache_ttl_seconds: int = 1800,
    ) -> None:
        self._stages: list[UrlFetchStage] = list(stages)
        self._cache = cache
        self._cache_ttl_seconds = cache_ttl_seconds

    async def fetch_one(
        self,
        url: str,
        *,
        identifiers: AcademicIdentifiers | None = None,
        client_sources: Sequence[ClientSourcePayload] = (),
        allowed_domains: tuple[str, ...] = (),
        blocked_domains: tuple[str, ...] = (),
    ) -> FetchResult:
        derived = extract_academic_identifiers(url)
        merged_ids = derived.merge(identifiers) if identifiers is not None else derived
        request = FetchRequest(
            url=url,
            identifiers=merged_ids,
            client_sources=tuple(client_sources),
            allowed_domains=allowed_domains,
            blocked_domains=blocked_domains,
        )

        # Client-supplied payloads bypass shared L2 cache because they reflect caller-scoped state.
        use_cache = self._cache is not None and self._cache_ttl_seconds > 0 and not client_sources
        cache_key = _fetch_cache_key(request)
        if use_cache and self._cache is not None:
            cached = await self._cache.get_json(cache_key)
            if cached is not None:
                return FetchResult.model_validate(cached)

        last_blocked: FetchResult | None = None
        for stage in self._stages:
            result = await stage.try_fetch(request)
            if result is None:
                continue
            if result.is_blocked:
                if result.blocked_reason in TERMINAL_SECURITY_BLOCKS:
                    return result
                last_blocked = result
                continue
            if result.markdown.strip():
                if use_cache and self._cache is not None:
                    await self._cache.set_json(
                        cache_key,
                        result.model_dump(),
                        ttl_seconds=self._cache_ttl_seconds,
                    )
                return result

        if last_blocked is not None:
            return last_blocked
        return FetchResult(
            url=url,
            final_url=url,
            stage_resolved="",
            is_blocked=True,
            blocked_reason="network_error",
            identifiers=merged_ids,
        )

    async def fetch_many(
        self,
        requests: Sequence[FetchRequest],
        *,
        concurrency: int = 3,
    ) -> tuple[FetchResult, ...]:
        if not requests:
            return ()
        sem = asyncio.Semaphore(max(1, concurrency))

        async def _guarded(req: FetchRequest) -> FetchResult:
            async with sem:
                return await self.fetch_one(
                    req.url,
                    identifiers=req.identifiers,
                    client_sources=req.client_sources,
                    allowed_domains=req.allowed_domains,
                    blocked_domains=req.blocked_domains,
                )

        return tuple(await asyncio.gather(*(_guarded(req) for req in requests)))


def build_default_fetch_ladder(
    config: WebSearchConfig,
    *,
    cache: SearchCache | None = None,
    allow_remote: bool = True,
    dns_resolver: DnsResolver | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> UrlFetchLadder:
    """Assemble default Escalation Ladder (Client IP -> Open Mirror -> Safe HTTP)."""
    stages: list[UrlFetchStage] = [ClientSuppliedFetchStage()]
    if allow_remote:
        stages.extend(
            (
                SafeHttpFetchStage(
                    timeout_seconds=config.timeout_seconds,
                    max_fetch_bytes=config.max_fetch_bytes,
                    user_agent=config.user_agent,
                    dns_resolver=dns_resolver,
                    http_client=http_client,
                ),
                OpenAcademicMirrorStage(
                    openalex_base_url=config.openalex_base_url,
                    openalex_api_key=config.openalex_api_key.get_secret_value(),
                    arxiv_base_url=config.arxiv_base_url,
                    europepmc_base_url=config.europepmc_base_url,
                    crossref_base_url=config.crossref_base_url,
                    timeout_seconds=config.timeout_seconds,
                    user_agent=config.user_agent,
                    http_client=http_client,
                ),
            )
        )
    return UrlFetchLadder(stages, cache=cache, cache_ttl_seconds=config.cache_ttl_seconds)
