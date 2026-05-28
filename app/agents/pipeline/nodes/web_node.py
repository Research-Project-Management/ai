"""
Web Research Node — Multi-source web search for the pipeline.

Search sources:
  - DuckDuckGo (general web, optionally filtered by academic sites)
  - OpenAlex (academic database via web_search_agent._search)
  - Tavily (if TAVILY_API_KEY configured — higher quality snippets)

All blocking search functions run in a thread pool to avoid blocking asyncio.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.agents.pipeline.state import PipelineState
from app.config import get_settings

logger = logging.getLogger(__name__)


def _search_tavily(
    query: str,
    api_key: str,
    include_domains: list[str] | None = None,
    max_results: int = 5,
) -> list[dict]:
    """Search using Tavily API (higher quality snippets than DDG)."""
    try:
        import httpx

        payload: dict[str, Any] = {
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        if include_domains:
            payload["include_domains"] = include_domains

        resp = httpx.post("https://api.tavily.com/search", json=payload, timeout=15)
        resp.raise_for_status()

        return [
            {
                "title": item.get("title", ""),
                "href": item.get("url", ""),
                "body": (item.get("content", ""))[:800],
                "source": "tavily",
            }
            for item in resp.json().get("results", [])
        ]
    except Exception as exc:
        logger.warning("Tavily API call failed: %s", exc)
        return []


def _format_web_context(results: list[dict], max_results: int = 12) -> str:
    """Format web search results into [P#] blocks for the LLM context."""
    lines = ["WEB RESEARCH RESULTS:"]
    for i, r in enumerate(results[:max_results], 1):
        title = r.get("title", "No title")
        href = r.get("href", r.get("url", ""))
        body = (r.get("body", r.get("content", "")))[:800]
        source_tag = "[OpenAlex]" if r.get("source") == "openalex" else "[Web]"

        lines.append(f"[P{i}] {source_tag}")
        lines.append(f"Title: {title}")
        if href:
            lines.append(f"URL: {href}")
        lines.append(f"TEXT: {body}")
        lines.append("")

    return "\n".join(lines).strip()


async def web_research_node(state: PipelineState) -> dict:
    """Search the web and return formatted context."""
    logger.info("════ Web Research Node ════")

    query_original = state.get("query_original", "")
    web_search_sites = state.get("web_search_sites", [])

    if not query_original:
        return {"web_context": "", "route": "done"}

    # Translate to English for search (async, cached)
    from app.agents.utils import translate_to_english_async

    query_en = state.get("query_en") or await translate_to_english_async(
        query_original
    )

    # ─── Run all search sources in PARALLEL ───────────────────────
    settings = get_settings()

    async def _run_ddg_openalex() -> list[dict]:
        """DDG + OpenAlex (internally parallelized via ThreadPoolExecutor)."""
        try:
            from app.agents.web_search_agent import _search

            return await asyncio.to_thread(
                _search,
                query_en,
                web_search_sites or None,
                pre_translated_query=query_en,
            )
        except Exception as exc:
            logger.warning("DDG/OpenAlex search failed: %s", exc)
            return []

    async def _run_tavily() -> list[dict]:
        """Tavily search (if API key configured)."""
        if not settings.tavily_api_key:
            return []
        try:
            return await asyncio.to_thread(
                _search_tavily,
                query_en,
                settings.tavily_api_key,
                web_search_sites,
            )
        except Exception as exc:
            logger.warning("Tavily search failed: %s", exc)
            return []

    # Fire both searches concurrently
    ddg_oalex_results, tavily_results = await asyncio.gather(
        _run_ddg_openalex(),
        _run_tavily(),
    )

    logger.info(
        "Parallel search done: DDG+OpenAlex=%d, Tavily=%d",
        len(ddg_oalex_results),
        len(tavily_results),
    )

    # ─── Merge + dedup Tavily results ─────────────────────────────
    results = ddg_oalex_results
    if tavily_results:
        seen_urls = {
            r.get("href", r.get("url", ""))
            for r in results
            if r.get("href") or r.get("url")
        }
        for tr in tavily_results:
            url = tr.get("href", tr.get("url", ""))
            if url and url not in seen_urls:
                seen_urls.add(url)
                results.append(tr)

    if not results:
        logger.info("No web results found")
        return {"web_context": "", "query_en": query_en, "route": "done"}

    # Format context
    web_context = _format_web_context(results)

    # Build sources for FE (merged with any existing RAG sources)
    existing_sources = state.get("sources", [])
    web_sources = [
        {
            "title": r.get("title", ""),
            "url": r.get("href", r.get("url", "")),
            "authors": r.get("authors") or None,
            "year": r.get("year") or None,
            "snippet": (r.get("body", r.get("content", "")))[:200] or None,
        }
        for r in results
    ]

    return {
        "web_context": web_context,
        "query_en": query_en,
        "route": "done",
        "sources": existing_sources + web_sources,
    }
