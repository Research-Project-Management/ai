"""LangGraph state definition isolated inside the `web_search_agent` subgraph."""

from typing_extensions import TypedDict

from ai.agents.web_search.fetch.contracts import ClientSourcePayload
from ai.agents.web_search.ranking.contracts import BlockedSourceReport, RankedWebEvidence
from ai.agents.web_search.search.contracts import SearchHit
from ai.agents.web_search.workflow.contracts import SearchMode
from ai.orchestrator.contracts import CitationItem
from ai.shared.contracts.run_control import RunUsage


class WebSearchState(TypedDict, total=False):
    """Subgraph state for planning, searching, fetching, evaluating, and citing."""

    query: str
    conversation_tail: list[str]
    seed_urls: list[str]
    allowed_domains: list[str]
    blocked_domains: list[str]
    client_sources: list[ClientSourcePayload]
    search_mode_override: SearchMode | None

    round_index: int
    standalone_query: str
    planned_mode: SearchMode
    sub_queries: list[str]
    raw_hits: list[SearchHit]
    repacked_evidence: list[RankedWebEvidence]
    ranked_desc: list[RankedWebEvidence]
    blocked_sources: list[BlockedSourceReport]
    abstain: bool
    usage: RunUsage
    final_answer: str
    citations: list[CitationItem]
