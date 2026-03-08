"""Retriever — semantic search over the vector store."""

from __future__ import annotations

import logging
from typing import Any

from app.rag.store import get_vector_store

logger = logging.getLogger(__name__)


async def retrieve_documents(
    query: str,
    k: int = 5,
    score_threshold: float = 0.3,
    document_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Perform semantic search and return relevant document chunks.

    If document_ids is provided, restricts search to only those documents.
    Returns a list of dicts with keys: content, source, score, metadata.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    store = get_vector_store()

    search_kwargs: dict[str, Any] = {}
    if document_ids:
        # Filter Qdrant to only the user's explicitly uploaded docs
        search_kwargs["filter"] = Filter(
            must=[
                FieldCondition(
                    key="metadata.doc_id",
                    match=MatchAny(any=document_ids),
                )
            ]
        )

    results = await store.asimilarity_search_with_score(query, k=k, **search_kwargs)

    documents = []
    for doc, score in results:
        # When filtering to specific docs, use a very permissive threshold —
        # the user explicitly uploaded these files so we always return something.
        effective_threshold = 0.0 if document_ids else score_threshold
        if score >= effective_threshold:
            documents.append(
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "score": round(float(score), 4),
                    "metadata": doc.metadata,
                }
            )

    logger.info("Retrieved %d documents for query: %s...", len(documents), query[:50])
    return documents
