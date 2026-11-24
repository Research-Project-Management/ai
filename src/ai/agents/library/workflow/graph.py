"""Subgraph compiler and Root Orchestrator AgentSpec adapter for `library_agent`."""

from __future__ import annotations

from typing import Any, Literal, cast

from langgraph.graph.state import CompiledStateGraph

from ai.agents.library.gateway.contracts import LibraryGateway
from ai.agents.library.tools.contracts import LibraryToolRegistry, build_default_tool_registry
from ai.agents.library.workflow.nodes import build_library_nodes
from ai.agents.library.workflow.state import LibraryState
from ai.orchestrator.contracts import AgentSpec
from ai.orchestrator.state import RootState
from ai.shared.contracts.sources import parse_attached_sources
from ai.shared.llm.factory import ChatRunnable
from ai.shared.workflow.events import ToolEventPublisher
from ai.shared.workflow.graph import build_tool_agent_spec, compile_tool_agent_graph
from ai.shared.workflow.plan import ToolAgentConfig

_VALID_SCOPES: frozenset[str] = frozenset({"auto", "personal", "project"})
_DEFAULT_LIBRARY_AGENT_CONFIG = ToolAgentConfig(max_steps=5, timeout_seconds=60.0)


def build_library_subgraph(
    llm: ChatRunnable,
    gateway: LibraryGateway,
    *,
    tool_registry: LibraryToolRegistry | None = None,
    config: ToolAgentConfig | None = None,
    event_publisher: ToolEventPublisher | None = None,
) -> CompiledStateGraph:
    """Compile the `plan_and_act -> policy_gate -> execute_and_verify -> synthesize_response` StateGraph."""
    cfg = config or _DEFAULT_LIBRARY_AGENT_CONFIG
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
    config: ToolAgentConfig | None = None,
    event_publisher: ToolEventPublisher | None = None,
) -> AgentSpec:
    """Build the pluggable `library_agent` AgentSpec with HITL-gated write actions."""
    cfg = config or _DEFAULT_LIBRARY_AGENT_CONFIG
    subgraph = build_library_subgraph(
        llm,
        gateway,
        tool_registry=tool_registry,
        config=cfg,
        event_publisher=event_publisher,
    )

    def _extra_state(
        state: RootState, context: dict[str, Any], configurable: dict[str, Any]
    ) -> dict[str, Any]:
        del state
        raw_scope = (
            str(context.get("scope_type") or configurable.get("scope_type") or "auto")
            .strip()
            .lower()
        )
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
        return {"scope_type": scope_type, "attached_items": attached_items}

    return build_tool_agent_spec(
        name="library_agent",
        description=(
            "Searches and summarizes structured papers/stats/collections/tags/notes in the user's Flux "
            "Library, checks duplicates/retractions, formats citations (APA/IEEE/BibTeX), adds papers "
            "by DOI/arXiv/URL, organizes items, and safely trashes or merges duplicates with confirmation."
        ),
        capabilities=(
            "library_search",
            "library_stats",
            "citations",
            "curation",
            "hitl_actions",
        ),
        timeout_seconds=cfg.timeout_seconds,
        subgraph=subgraph,
        empty_query_message="Yêu cầu thư viện đang trống.",
        build_extra_state=_extra_state,
    )
