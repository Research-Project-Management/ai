"""Internal LangGraph state definition for the `library_agent` subgraph."""

from typing import Any, Literal

from typing_extensions import TypedDict

from ai.orchestrator.contracts import CitationItem
from ai.shared.contracts.run_control import RunUsage


class LibraryState(TypedDict, total=False):
    """JSON/msgpack-safe state passing through `plan_action -> policy_gate -> execute_tools -> synthesize_response`."""

    query: str
    conversation_tail: list[str]
    space_id: str
    user_id: str
    scope_type: Literal["auto", "personal", "project"]
    attached_items: list[dict[str, Any]]
    step: int
    pending_calls: list[dict[str, Any]]
    approved_calls: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    citations: list[CitationItem]
    draft_answer: str
    final_answer: str
    usage: RunUsage
    done: bool
