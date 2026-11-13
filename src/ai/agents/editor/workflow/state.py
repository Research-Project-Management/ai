"""Internal LangGraph state definition for the `editor_agent` subgraph."""

from typing import Any

from typing_extensions import TypedDict

from ai.orchestrator.contracts import CitationItem
from ai.shared.contracts.run_control import RunUsage


class EditorState(TypedDict, total=False):
    """JSON/msgpack-safe state passing through the 5-node `editor_agent` workflow."""

    query: str
    conversation_tail: list[str]
    peer_findings: list[dict[str, Any]]
    page_id: str
    space_id: str
    user_id: str
    raw_editor_context: dict[str, Any]
    context_snapshot_text: str
    active_file_id: str | None
    active_file_path: str
    file_cache: dict[str, dict[str, Any]]
    known_cite_keys: list[str]
    known_labels: list[str]
    step: int
    pending_calls: list[dict[str, Any]]
    approved_calls: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    patches: list[dict[str, Any]]
    citations: list[CitationItem]
    draft_answer: str
    final_answer: str
    usage: RunUsage
    done: bool
