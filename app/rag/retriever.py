"""Retriever — semantic search over Qdrant."""

from __future__ import annotations

import logging
from typing import Any

from app.rag.store import get_vector_store

logger = logging.getLogger(__name__)


async def retrieve_documents(
    query: str,
    k: int = 8,
    score_threshold: float = 0.3,
    document_ids: list[str] | None = None,
    user_id: str | None = None,
    chat_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Similarity search over the vector store.

    Args:
        query: The search query.  Pass the best English representation for
               good embedding alignment.  The caller is responsible for
               choosing an appropriate query strategy (targeted vs. broad
               coverage) and k.
        k: Maximum number of chunks to return.
        score_threshold: Minimum cosine similarity.  Ignored when
                         document_ids is provided (user explicitly pinned
                         these files so we return everything).
        document_ids: Restrict search to these Qdrant doc IDs.
        user_id: Only return chunks uploaded by this user (RAG isolation).
        chat_id: Only return chunks uploaded in this chat session (RAG isolation).

    Returns:
        List of dicts: {content, source, score, metadata}
    """
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

    store = get_vector_store()

    filter_conditions = []

    # Scope to specific document IDs if requested
    if document_ids:
        filter_conditions.append(
            FieldCondition(
                key="metadata.doc_id",
                match=MatchAny(any=document_ids),
            )
        )

    # Always enforce user + chat isolation when provided
    if user_id:
        filter_conditions.append(
            FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))
        )
    if chat_id:
        filter_conditions.append(
            FieldCondition(key="metadata.chat_id", match=MatchValue(value=chat_id))
        )

    search_kwargs: dict[str, Any] = {}
    if filter_conditions:
        search_kwargs["filter"] = Filter(must=filter_conditions)

    results = await store.asimilarity_search_with_score(query, k=k, **search_kwargs)

    chunks = []
    for doc, score in results:
        effective_threshold = 0.0 if document_ids else score_threshold
        if score >= effective_threshold:
            chunks.append(
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "score": round(float(score), 4),
                    "metadata": doc.metadata,
                }
            )

    logger.info("retrieve_documents: query=%r k=%d → %d chunks", query[:60], k, len(chunks))
    return chunks

