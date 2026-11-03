"""Subgraph compiler and Root Orchestrator AgentSpec adapter for `library_agent`."""

from __future__ import annotations

from typing import Any, Literal, cast

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from ai.agents.library.gateway.contracts import LibraryGateway
from ai.agents.library.tools.contracts import LibraryToolRegistry
from ai.agents.library.tools.registry import build_default_tool_registry
from ai.agents.library.workflow.contracts import LibraryAgentConfig
from ai.agents.library.workflow.nodes import build_library_nodes
from ai.agents.library.workflow.state import LibraryState
from ai.orchestrator.contracts import AgentOutput, AgentSpec
from ai.orchestrator.state import RootState
from ai.shared.contracts.run_control import RunUsage
from ai.shared.contracts.sources import parse_attached_sources
from ai.shared.llm.factory import ChatRunnable
from ai.shared.workflow.events import ToolEventPublisher
from ai.shared.workflow.graph import compile_tool_agent_graph
from ai.shared.workflow.turn import latest_query, recent_tail

_VALID_SCOPES: frozenset[str] = frozenset({"auto", "personal", "project"})


def build_library_subgraph(
    llm: ChatRunnable,
    gateway: LibraryGateway,
    *,
    tool_registry: LibraryToolRegistry | None = None,
    config: LibraryAgentConfig | None = None,
    event_publisher: ToolEventPublisher | None = None,
) -> CompiledStateGraph:
    """Compile the `plan_and_act -> policy_gate -> execute_and_verify -> synthesize_response` StateGraph."""
    cfg = config or LibraryAgentConfig()
    registry = tool_registry or build_default_tool_registry()
    nodes = build_library_nodes(
        llm=llm,
        gateway=gateway,
        tool_registry=registry,
        agent_config=cfg,
        event_publisher=event_publisher,
    )

    return compile_tool_agent_graph(LibraryState, nodes, max_steps=cfg.max_steps)


def build_library_agent_spec(
    llm: ChatRunnable,
    gateway: LibraryGateway,
    *,
    tool_registry: LibraryToolRegistry | None = None,
    config: LibraryAgentConfig | None = None,
    event_publisher: ToolEventPublisher | None = None,
) -> AgentSpec:
    """Build the pluggable `library_agent` AgentSpec with HITL-gated write actions."""
    cfg = config or LibraryAgentConfig()
    subgraph = build_library_subgraph(
        llm,
        gateway,
        tool_registry=tool_registry,
        config=cfg,
        event_publisher=event_publisher,
    )

    async def _run_library_agent(state: RootState, run_config: RunnableConfig) -> AgentOutput:
        configurable: dict[str, Any] = dict(run_config.get("configurable") or {})
        context: dict[str, Any] = dict(state.get("context") or {})

        query = latest_query(state)
        if not query:
            return AgentOutput(
                agent="library_agent",
                content="Yêu cầu thư viện đang trống.",
                status="error",
                error="empty query",
            )

        space_id = str(configurable.get("space_id") or context.get("space_id") or "").strip()
        user_id = str(context.get("user_id") or configurable.get("user_id") or "").strip()
        raw_scope = str(context.get("scope_type") or configurable.get("scope_type") or "auto").strip().lower()
        scope_type: Literal["auto", "personal", "project"] = cast(
            Literal["auto", "personal", "project"],
            raw_scope if raw_scope in _VALID_SCOPES else "auto",
        )
        attached = parse_attached_sources(context.get("attached_sources"))
        attached_items = [
            {
                "item_id": src.item_id,
                "title": src.title or "Untitled",
                "has_pdf": bool(src.has_pdf or src.file_id),
                "indexed": bool(src.indexed),
            }
            for src in attached
            if src.kind in ("library", "paper") and src.item_id
        ]

        sub_input: LibraryState = {
            "query": query,
            "conversation_tail": recent_tail(list(state.get("messages") or ())),
            "space_id": space_id,
            "user_id": user_id,
            "scope_type": scope_type,
            "attached_items": attached_items,
            "step": 0,
            "pending_calls": [],
            "approved_calls": [],
            "observations": [],
            "citations": [],
            "draft_answer": "",
            "done": False,
        }
        result = await subgraph.ainvoke(sub_input, config=run_config)
        observations = result.get("observations") or ()
        raw_usage = result.get("usage")
        return AgentOutput(
            agent="library_agent",
            content=str(result.get("final_answer") or ""),
            status="ok",
            citations=tuple(result.get("citations") or ()),
            artifacts={
                "steps": int(result.get("step", 1)),
                "tool_calls_executed": len(observations),
            },
            usage=raw_usage if isinstance(raw_usage, RunUsage) else RunUsage(),
            is_final=True,
        )

    return AgentSpec(
        name="library_agent",
        description=(
            "Searches and summarizes structured papers/stats/collections/tags/notes in the user's Flux "
            "Library, checks duplicates/retractions, formats citations (APA/IEEE/BibTeX), adds papers "
            "by DOI/arXiv/URL, organizes items, and safely trashes or merges duplicates with confirmation."
        ),
        handler=_run_library_agent,
        capabilities=(
            "library_search",
            "library_stats",
            "citations",
            "curation",
            "hitl_actions",
        ),
        timeout_seconds=cfg.timeout_seconds,
    )
