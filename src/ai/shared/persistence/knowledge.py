"""Space-isolated Hybrid (Dense + Sparse RRF) KnowledgeStore backed by Qdrant."""

import asyncio
import math
import re
import uuid
import zlib
from collections import Counter
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

from qdrant_client import AsyncQdrantClient, models

from ai.shared.config.embedding import EmbeddingConfig
from ai.shared.config.llm import LLMConfig
from ai.shared.contracts.knowledge import (
    Embedder,
    KnowledgeChunkInput,
    KnowledgeDocumentInput,
    KnowledgeHit,
    KnowledgeStore,
)
from ai.shared.llm.embeddings import build_embedder

_TOKEN_RE = re.compile(r"\w+")
_SPARSE_BUCKETS = 2_000_003


def chunk_text(
    text: str,
    *,
    max_chars: int = 1200,
    overlap: int = 150,
) -> list[KnowledgeChunkInput]:
    """Split text into bounded overlapping chunks along natural boundaries."""
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [KnowledgeChunkInput(chunk_index=0, content=cleaned)]

    chunks: list[KnowledgeChunkInput] = []
    start = 0
    idx = 0
    total = len(cleaned)
    while start < total:
        end = min(total, start + max_chars)
        if end < total:
            split_newline = cleaned.rfind("\n", start + max_chars // 2, end)
            split_space = cleaned.rfind(" ", start + max_chars // 2, end)
            boundary = max(split_newline, split_space)
            if boundary > start:
                end = boundary

        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(KnowledgeChunkInput(chunk_index=idx, content=piece))
            idx += 1

        if end >= total:
            break
        start = max(end - overlap, start + 1)

    return chunks


def compute_sparse_vector(text: str) -> models.SparseVector:
    """Compute a hashed BM25-style sublinear TF sparse vector."""
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return models.SparseVector(indices=[], values=[])

    counts = Counter(tokens)
    buckets: dict[int, float] = {}
    for token, count in counts.items():
        bucket = zlib.crc32(token.encode("utf-8")) % _SPARSE_BUCKETS
        buckets[bucket] = buckets.get(bucket, 0.0) + (1.0 + math.log(count))

    ordered_indices = sorted(buckets.keys())
    return models.SparseVector(
        indices=ordered_indices,
        values=[buckets[idx] for idx in ordered_indices],
    )


class QdrantKnowledgeStore:
    """Idempotent Qdrant vector store enforcing hard `space_id` multi-tenant filters."""

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        embedder: Embedder,
        *,
        collection_name: str = "kb_bge_m3",
        collection_alias: str = "",
    ) -> None:
        self._qdrant = qdrant_client
        self._embedder = embedder
        self._collection = collection_name
        self._alias = collection_alias.strip()
        self._target = self._alias or self._collection
        self._collection_ready = False
        self._init_lock = asyncio.Lock()

    async def _ensure_alias(self) -> None:
        if not self._alias or self._alias == self._collection:
            return
        aliases_res = await self._qdrant.get_aliases()
        existing = next(
            (item for item in aliases_res.aliases if item.alias_name == self._alias),
            None,
        )
        if existing is not None and existing.collection_name == self._collection:
            return

        ops: list[models.AliasOperations] = []
        if existing is not None:
            ops.append(
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=self._alias))
            )
        ops.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=self._collection,
                    alias_name=self._alias,
                )
            )
        )
        await self._qdrant.update_collection_aliases(change_aliases_operations=ops)

    async def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        async with self._init_lock:
            if self._collection_ready:
                return
            exists = await self._qdrant.collection_exists(self._collection)
            if not exists:
                await self._qdrant.create_collection(
                    collection_name=self._collection,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=self._embedder.dimensions,
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={"sparse": models.SparseVectorParams()},
                )
            await self._ensure_alias()
            self._collection_ready = True

    async def ensure_ready(self) -> None:
        """Eagerly verify/create the target collection and alias before serving traffic."""
        await self._ensure_collection()

    async def upsert(self, doc: KnowledgeDocumentInput) -> int:
        chunks = list(doc.chunks) if doc.chunks else chunk_text(doc.content)
        if not chunks:
            raise ValueError("KnowledgeDocumentInput must provide non-empty content or chunks")

        await self._ensure_collection()
        await self.delete_source(
            space_id=doc.space_id,
            source_type=doc.source_type,
            source_id=doc.source_id,
        )

        dense_vectors = await self._embedder.embed_documents([c.content for c in chunks])
        points: list[models.PointStruct] = []
        for chunk, dense_vec in zip(chunks, dense_vectors, strict=True):
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{doc.space_id}:{doc.source_type}:{doc.source_id}:{chunk.chunk_index}",
                )
            )
            sparse_vec = compute_sparse_vector(chunk.content)
            payload = {
                "space_id": doc.space_id,
                "source_type": doc.source_type,
                "source_id": doc.source_id,
                "title": doc.title,
                "url": doc.url,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "metadata": {**doc.metadata, **chunk.metadata},
            }
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={"dense": dense_vec, "sparse": sparse_vec},
                    payload=payload,
                )
            )

        await self._qdrant.upsert(collection_name=self._target, points=points)
        return len(points)

    async def has_source(self, *, space_id: str, source_id: str) -> bool:
        """Return True when at least one chunk exists for `source_id` in `space_id`."""
        await self._ensure_collection()
        res = await self._qdrant.count(
            collection_name=self._target,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="space_id",
                        match=models.MatchValue(value=space_id),
                    ),
                    models.FieldCondition(
                        key="source_id",
                        match=models.MatchValue(value=source_id),
                    ),
                ]
            ),
            exact=False,
        )
        return res.count > 0

    async def search(
        self,
        *,
        space_id: str,
        query: str,
        limit: int = 5,
        source_types: Sequence[str] = (),
        source_ids: Sequence[str] = (),
    ) -> list[KnowledgeHit]:
        await self._ensure_collection()
        must: list[models.Condition] = [
            models.FieldCondition(
                key="space_id",
                match=models.MatchValue(value=space_id),
            )
        ]
        if source_types:
            must.append(
                models.FieldCondition(
                    key="source_type",
                    match=models.MatchAny(any=list(source_types)),
                )
            )
        if source_ids:
            must.append(
                models.FieldCondition(
                    key="source_id",
                    match=models.MatchAny(any=list(source_ids)),
                )
            )

        q_filter = models.Filter(must=must)
        dense_query = await self._embedder.embed_query(query)
        sparse_query = compute_sparse_vector(query)

        if sparse_query.indices:
            res = await self._qdrant.query_points(
                collection_name=self._target,
                prefetch=[
                    models.Prefetch(
                        query=dense_query,
                        using="dense",
                        limit=max(limit * 2, 10),
                    ),
                    models.Prefetch(
                        query=sparse_query,
                        using="sparse",
                        limit=max(limit * 2, 10),
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=q_filter,
                limit=limit,
                with_payload=True,
            )
        else:
            res = await self._qdrant.query_points(
                collection_name=self._target,
                query=dense_query,
                using="dense",
                query_filter=q_filter,
                limit=limit,
                with_payload=True,
            )

        hits: list[KnowledgeHit] = []
        for point in res.points:
            payload = point.payload or {}
            hits.append(
                KnowledgeHit(
                    source_id=str(payload.get("source_id", "")),
                    source_type=str(payload.get("source_type", "")),
                    title=str(payload.get("title", "")),
                    content=str(payload.get("content", "")),
                    score=float(point.score),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    page=payload.get("page"),
                    url=payload.get("url"),
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
        return hits

    async def delete_source(self, *, space_id: str, source_type: str, source_id: str) -> bool:
        await self._ensure_collection()
        await self._qdrant.delete(
            collection_name=self._target,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="space_id",
                            match=models.MatchValue(value=space_id),
                        ),
                        models.FieldCondition(
                            key="source_type",
                            match=models.MatchValue(value=source_type),
                        ),
                        models.FieldCondition(
                            key="source_id",
                            match=models.MatchValue(value=source_id),
                        ),
                    ]
                )
            ),
        )
        return True

    async def delete_space(self, *, space_id: str) -> bool:
        await self._ensure_collection()
        await self._qdrant.delete(
            collection_name=self._target,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="space_id",
                            match=models.MatchValue(value=space_id),
                        )
                    ]
                )
            ),
        )
        return True


@asynccontextmanager
async def create_knowledge_store(
    embedding_cfg: EmbeddingConfig,
    llm_cfg: LLMConfig,
    qdrant_client: AsyncQdrantClient | None,
    *,
    embedder: Embedder | None = None,
    allow_remote_embeddings: bool = True,
) -> AsyncIterator[KnowledgeStore]:
    """Yield a QdrantKnowledgeStore backed by remote Qdrant or local in-memory Qdrant."""
    active_embedder = embedder or build_embedder(
        embedding_cfg,
        llm_cfg,
        allow_remote=allow_remote_embeddings,
    )
    if qdrant_client is not None:
        yield QdrantKnowledgeStore(
            qdrant_client,
            active_embedder,
            collection_name=embedding_cfg.collection_name,
            collection_alias=embedding_cfg.collection_alias,
        )
        return

    local_client = AsyncQdrantClient(location=":memory:")
    try:
        yield QdrantKnowledgeStore(
            local_client,
            active_embedder,
            collection_name=embedding_cfg.collection_name,
            collection_alias=embedding_cfg.collection_alias,
        )
    finally:
        await local_client.close()
