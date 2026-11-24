"""Internal LangGraph state definition for the `rag_agent` subgraph."""

from typing import Literal

from typing_extensions import TypedDict

from ai.agents.rag.retrieval.contracts import RankedPassage
from ai.orchestrator.contracts import CitationItem
from ai.shared.contracts.run_control import RunUsage


class RagState(TypedDict, total=False):
    """State passing through `transform_query -> retrieve -> evaluate_crag -> generate_answer`."""

    query: str
    standalone_query: str
    conversation_tail: list[str]
    space_id: str
    source_ids: tuple[str, ...]
    source_types: tuple[str, ...]
    attempt: int
    search_queries: list[str]
    passages: list[RankedPassage]
    ranked_desc: list[RankedPassage]
    total_candidates: int
    abstain: bool
    next_route: Literal["retry", "answer"]
    usage: RunUsage
    final_answer: str
    citations: list[CitationItem]
