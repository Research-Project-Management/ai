"""Internal LangGraph JSON/msgpack-safe state definition for `project_agent`."""

from typing import Any

from typing_extensions import TypedDict

from ai.orchestrator.contracts import CitationItem
from ai.shared.contracts.run_control import RunUsage


class ProjectState(TypedDict, total=False):
    """Typed state flowing through the 5-node `project_agent` StateGraph."""

    query: str
    conversation_tail: list[str]
    peer_findings: list[dict[str, Any]]
    space_id: str
    project_id: str | None
    user_id: str
    raw_context: dict[str, Any]
    context_snapshot_text: str
    projects_cache: list[dict[str, Any]]
    states_cache: dict[str, list[dict[str, Any]]]
    members_cache: dict[str, list[dict[str, Any]]]
    labels_cache: dict[str, list[dict[str, Any]]]
    step: int
    pending_calls: list[dict[str, Any]]
    approved_calls: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    widgets: list[dict[str, Any]]
    citations: list[CitationItem]
    draft_answer: str
    final_answer: str
    usage: RunUsage
    done: bool
