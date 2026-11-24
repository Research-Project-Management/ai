"""Subgraph compiler and Root Orchestrator AgentSpec adapter for `web_search_agent`."""

from collections.abc import Sequence
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from ai.agents.web_search.fetch.contracts import ClientSourcePayload
from ai.agents.web_search.fetch.extractor import extract_urls_from_text
from ai.agents.web_search.fetch.ladder import UrlFetchLadder
from ai.agents.web_search.search.contracts import normalize_domain_entry
from ai.agents.web_search.search.registry import SearchProviderRegistry
from ai.agents.web_search.workflow.contracts import VALID_MODES, SearchMode, WebSearchAgentConfig
from ai.agents.web_search.workflow.nodes import build_web_search_nodes
from ai.agents.web_search.workflow.state import WebSearchState
from ai.orchestrator.contracts import AgentOutput, AgentSpec
from ai.orchestrator.state import RootState
from ai.shared.contracts.run_control import RunUsage
from ai.shared.llm.factory import ChatRunnable
from ai.shared.workflow.turn import latest_query, recent_tail


def _extract_string_list(primary: Any, secondary: Any = None) -> list[str]:
    source = primary if primary else secondary
    if isinstance(source, str):
        return [s.strip() for s in source.split(",") if s.strip()]
    if isinstance(source, Sequence):
        return [str(item).strip() for item in source if str(item or "").strip()]
    return []


def _parse_client_sources(raw: Any) -> list[ClientSourcePayload]:
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        return []
    parsed: list[ClientSourcePayload] = []
    for item in raw:
        if isinstance(item, ClientSourcePayload):
            parsed.append(item)
        elif isinstance(item, dict):
            parsed.append(ClientSourcePayload.model_validate(item))
    return parsed


def build_web_search_subgraph(
    llm: ChatRunnable,
    search_registry: SearchProviderRegistry,
    fetch_ladder: UrlFetchLadder,
    *,
    config: WebSearchAgentConfig | None = None,
) -> CompiledStateGraph:
    """Compile the 4-node iterative external search & safe-fetch StateGraph."""
    cfg = config or WebSearchAgentConfig()
    nodes = build_web_search_nodes(
        llm=llm,
        search_registry=search_registry,
        fetch_ladder=fetch_ladder,
        agent_config=cfg,
    )

    def _route_after_eval(state: WebSearchState) -> str:
        return state.get("next_route", "answer")

    builder = StateGraph(WebSearchState)
    builder.add_node("plan_search", nodes["plan_search"])
    builder.add_node("execute_search_and_fetch", nodes["execute_search_and_fetch"])
    builder.add_node("evaluate_coverage", nodes["evaluate_coverage"])
    builder.add_node("synthesize_answer", nodes["synthesize_answer"])

    builder.add_edge(START, "plan_search")
    builder.add_edge("plan_search", "execute_search_and_fetch")
    builder.add_edge("execute_search_and_fetch", "evaluate_coverage")
    builder.add_conditional_edges(
        "evaluate_coverage",
        _route_after_eval,
        {"retry": "plan_search", "answer": "synthesize_answer"},
    )
    builder.add_edge("synthesize_answer", END)
    return builder.compile()


def build_web_search_agent_spec(
    llm: ChatRunnable,
    search_registry: SearchProviderRegistry,
    fetch_ladder: UrlFetchLadder,
    *,
    config: WebSearchAgentConfig | None = None,
) -> AgentSpec:
    """Build the pluggable `web_search_agent` AgentSpec isolated from internal Space docs."""
    cfg = config or WebSearchAgentConfig()
    subgraph = build_web_search_subgraph(llm, search_registry, fetch_ladder, config=cfg)

    async def _run_web_search_agent(state: RootState, run_config: RunnableConfig) -> AgentOutput:
        configurable: dict[str, Any] = dict(run_config.get("configurable") or {})
        context: dict[str, Any] = dict(state.get("context") or {})

        query = latest_query(state)
        if not query:
            return AgentOutput(
                agent="web_search_agent",
                content="Truy vấn tìm kiếm bên ngoài đang trống.",
                status="error",
                error="empty query",
            )

        raw_allowed = _extract_string_list(
            context.get("allowed_domains") or context.get("include_domains"),
            configurable.get("allowed_domains") or configurable.get("include_domains"),
        )
        raw_blocked = _extract_string_list(
            context.get("blocked_domains") or context.get("exclude_domains"),
            configurable.get("blocked_domains") or configurable.get("exclude_domains"),
        )
        try:
            allowed_domains = [normalize_domain_entry(d) for d in raw_allowed]
            blocked_domains = [normalize_domain_entry(d) for d in raw_blocked]
            if allowed_domains and blocked_domains:
                raise ValueError(
                    "Cannot specify both allowed_domains and blocked_domains simultaneously"
                )
            client_sources = _parse_client_sources(
                context.get("external_sources") or context.get("client_sources")
            )
        except (ValueError, ValidationError) as exc:
            return AgentOutput(
                agent="web_search_agent",
                content=f"Tham số phạm vi nguồn ngoài không hợp lệ: {exc}",
                status="error",
                error=str(exc),
            )

        explicit_urls = _extract_string_list(
            context.get("target_urls") or context.get("urls"),
            configurable.get("target_urls") or configurable.get("urls"),
        )
        inline_urls = list(extract_urls_from_text(query))
        seen_urls: set[str] = set()
        seed_urls: list[str] = []
        for url in [*explicit_urls, *inline_urls]:
            if url not in seen_urls:
                seen_urls.add(url)
                seed_urls.append(url)

        raw_mode = (
            str(context.get("search_mode") or configurable.get("search_mode") or "").strip().lower()
        )
        mode_override: SearchMode | None = (
            cast(SearchMode, raw_mode) if raw_mode in VALID_MODES else None
        )

        # Air-Gap guarantee: pass ONLY recent user/assistant prompt turns & explicit external context,
        # never private Qdrant passages or Space document payloads.
        sub_input: WebSearchState = {
            "query": query,
            "conversation_tail": recent_tail(list(state.get("messages") or ())),
            "seed_urls": seed_urls,
            "allowed_domains": allowed_domains,
            "blocked_domains": blocked_domains,
            "client_sources": client_sources,
            "search_mode_override": mode_override,
            "round_index": 0,
        }
        result = await subgraph.ainvoke(sub_input, config=run_config)
        blocked_items = [item.model_dump() for item in (result.get("blocked_sources") or ())]
        raw_usage = result.get("usage")
        return AgentOutput(
            agent="web_search_agent",
            content=str(result.get("final_answer") or ""),
            status="ok",
            citations=tuple(result.get("citations") or ()),
            artifacts={
                "abstained": bool(result.get("abstain", False)),
                "rounds": int(result.get("round_index", 1)),
                "mode": str(result.get("planned_mode") or "hybrid"),
                "total_raw_hits": len(result.get("raw_hits") or ()),
                "blocked_sources": blocked_items,
            },
            usage=raw_usage if isinstance(raw_usage, RunUsage) else RunUsage(),
            is_final=True,
        )

    return AgentSpec(
        name="web_search_agent",
        description=(
            "Searches the public internet, academic repositories (OpenAlex, arXiv, Crossref), "
            "and caller-provided URLs or scoped domains with SSRF-safe fetching and verified URL/DOI citations."
        ),
        handler=_run_web_search_agent,
        capabilities=(
            "web_search",
            "scholar_search",
            "safe_url_fetch",
            "domain_scoping",
            "client_ip_sources",
            "grounded_citations",
        ),
        timeout_seconds=cfg.timeout_seconds,
    )
