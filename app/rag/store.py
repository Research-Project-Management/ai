"""Qdrant vector store wrapper using LangChain integration."""

from __future__ import annotations

import logging
from typing import Any

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import get_settings
from app.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None
_vector_store: QdrantVectorStore | None = None


def get_qdrant_client() -> QdrantClient:
    """Get or create the Qdrant client singleton."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        logger.info("Connected to Qdrant at %s", settings.qdrant_url)
    return _client


EMBEDDING_DIM = 3072  # gemini-embedding-001 (3072-dim output)


def _ensure_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    """Create payload indexes required for filtered searches.

    Qdrant Cloud requires explicit keyword/UUID indexes before a field can
    be used in a FieldCondition filter.  This function is idempotent — calling
    it on an existing index is a no-op.
    """
    from qdrant_client.models import PayloadSchemaType

    indexes = {
        "metadata.doc_id": PayloadSchemaType.KEYWORD,
        "metadata.source": PayloadSchemaType.KEYWORD,
    }
    existing = {}
    try:
        info = client.get_collection(collection_name)
        existing = info.payload_schema or {}
    except Exception:
        pass

    for field, schema in indexes.items():
        if field not in existing:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=schema,
                )
                logger.info("Created payload index on '%s'", field)
            except Exception as exc:
                logger.warning("Could not create index for '%s': %s", field, exc)


def get_vector_store() -> QdrantVectorStore:
    """Get or create the LangChain QdrantVectorStore singleton."""
    global _vector_store
    if _vector_store is None:
        settings = get_settings()
        embeddings = get_embeddings()
        client = get_qdrant_client()

        from qdrant_client.models import Distance, VectorParams

        collections = [c.name for c in client.get_collections().collections]
        if settings.qdrant_collection in collections:
            # Drop collection if vector size doesn't match (e.g. after model change)
            info = client.get_collection(settings.qdrant_collection)
            existing_size = info.config.params.vectors.size  # type: ignore[union-attr]
            if existing_size != EMBEDDING_DIM:
                logger.warning(
                    "Collection '%s' has %d-dim vectors but embeddings are %d-dim. Recreating.",
                    settings.qdrant_collection,
                    existing_size,
                    EMBEDDING_DIM,
                )
                client.delete_collection(settings.qdrant_collection)
                collections = []

        if settings.qdrant_collection not in collections:
            client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            logger.info("Created collection: %s", settings.qdrant_collection)

        # Ensure payload indexes exist (required for filtered searches on Qdrant Cloud)
        _ensure_payload_indexes(client, settings.qdrant_collection)

        _vector_store = QdrantVectorStore(
            client=client,
            collection_name=settings.qdrant_collection,
            embedding=embeddings,
        )
    return _vector_store


async def check_qdrant_health() -> bool:
    """Check if Qdrant is reachable."""
    try:
        client = get_qdrant_client()
        client.get_collections()
        return True
    except Exception:
        return False
