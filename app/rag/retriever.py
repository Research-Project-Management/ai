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

    Returns:
        List of dicts: {content, source, score, metadata}
    """
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    store = get_vector_store()

    search_kwargs: dict[str, Any] = {}
    if document_ids:
        search_kwargs["filter"] = Filter(
            must=[
                FieldCondition(
                    key="metadata.doc_id",
                    match=MatchAny(any=document_ids),
                )
            ]
        )

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
