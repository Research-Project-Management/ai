"""Pipeline state definition — shared TypedDict flowing through all graph nodes."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage


class PipelineState(TypedDict, total=False):
    """State that flows through the Supervisor Pipeline.

    Each node reads what it needs and returns ONLY the fields it modifies.
    Unmentioned fields are preserved automatically by LangGraph.
    """

    # Input
    messages: list[BaseMessage]

    # Routing
    route: str   # Current routing decision
    intent: str  # Final resolved intent for system prompt

    # Context (produced by worker nodes)
    rag_context: str         # Formatted [D#] blocks from RAG retrieval
    web_context: str         # Formatted [P#] blocks from web research
    sources: list[dict[str, Any]]  # Structured sources for FE META event

    # Query processing
    query_original: str  # Original user query (may be Vietnamese)
    query_en: str        # English translation (for search/rerank)

    # Configuration (set at pipeline start, read-only during execution)
    web_search_enabled: bool
    document_ids: list[str]
    web_search_sites: list[str]
    metadata: dict[str, Any]
    user_id: str   # Authenticated user — used for RAG isolation
    chat_id: str   # Current chat session — used for RAG isolation

    # Quick replies (greeting/end — chat.py sends this directly)
    early_reply: str
