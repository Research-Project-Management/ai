"""Unit tests for dynamic multi-document top-k scaling and stratified passage selection."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from ai.agents.rag.retrieval.contracts import RankedPassage, RetrievalQuery
from ai.agents.rag.retrieval.engine import (
    _MAX_MULTI_DOC_TOP_K,
    RetrievalEngine,
    _compute_effective_top_k,
)
from ai.shared.contracts.knowledge import KnowledgeDocumentInput, KnowledgeHit


class _FakeKnowledgeStore:
    def __init__(self, hits: list[KnowledgeHit]) -> None:
        self._hits = hits

    async def upsert(self, doc: KnowledgeDocumentInput) -> int:
        del doc
        return 0

    async def has_source(self, *, space_id: str, source_id: str) -> bool:
        del space_id, source_id
        return True

    async def search(
        self,
        *,
        space_id: str,
        query: str,
        limit: int = 5,
        source_types: Sequence[str] = (),
        source_ids: Sequence[str] = (),
    ) -> list[KnowledgeHit]:
        del space_id, query, source_types
        allowed = set(source_ids)
        filtered = [h for h in self._hits if not allowed or h.source_id in allowed]
        return filtered[:limit]

    async def delete_source(self, *, space_id: str, source_type: str, source_id: str) -> bool:
        del space_id, source_type, source_id
        return True

    async def delete_space(self, *, space_id: str) -> bool:
        del space_id
        return True


class _FixedScoreReranker:
    def __init__(self, score_map: dict[tuple[str, int], float]) -> None:
        self._score_map = score_map

    async def rerank(
        self,
        query: str,
        candidates: Sequence[KnowledgeHit],
        *,
        top_k: int,
    ) -> list[RankedPassage]:
        del query
        passages = [
            RankedPassage(
                source_id=hit.source_id,
                source_type=hit.source_type,
                title=hit.title,
                content=hit.content,
                rerank_score=self._score_map[(hit.source_id, hit.chunk_index)],
                retrieval_score=hit.score,
                chunk_index=hit.chunk_index,
                page=hit.page,
                url=hit.url,
                metadata=hit.metadata,
            )
            for hit in candidates
        ]
        passages.sort(key=lambda item: item.rerank_score, reverse=True)
        return passages[:top_k]


@pytest.mark.parametrize(
    ("scoped_doc_count", "expected"),
    [
        (0, 6),
        (1, 6),
        (2, 6),
        (3, 6),
        (4, 8),
        (5, 10),
        (8, 16),
        (12, _MAX_MULTI_DOC_TOP_K),
    ],
)
def test_compute_effective_top_k_scales_with_budget_cap(
    scoped_doc_count: int,
    expected: int,
) -> None:
    assert _compute_effective_top_k(6, scoped_doc_count) == expected


async def test_retrieve_stratifies_across_multiple_scoped_sources() -> None:
    hits = [
        *[
            KnowledgeHit(
                source_id="doc-a",
                source_type="paper",
                title="Paper A",
                content=f"Paper A method detail {idx}",
                score=0.95 - idx * 0.02,
                chunk_index=idx,
            )
            for idx in range(6)
        ],
        KnowledgeHit(
            source_id="doc-b",
            source_type="paper",
            title="Paper B",
            content="Paper B key comparison metric",
            score=0.40,
            chunk_index=0,
        ),
        KnowledgeHit(
            source_id="doc-c",
            source_type="paper",
            title="Paper C",
            content="Paper C ablation baseline",
            score=0.35,
            chunk_index=0,
        ),
    ]
    score_map = {(h.source_id, h.chunk_index): h.score for h in hits}
    engine = RetrievalEngine(
        knowledge_store=_FakeKnowledgeStore(hits),
        reranker=_FixedScoreReranker(score_map),
        default_top_k=6,
        default_min_score=0.05,
    )

    batch = await engine.retrieve(
        RetrievalQuery(
            space_id="sp-1",
            primary_query="compare methods across all three papers",
            source_ids=("doc-a", "doc-b", "doc-c"),
        )
    )

    assert len(batch.ranked_desc) == 6
    assert {p.source_id for p in batch.ranked_desc} == {"doc-a", "doc-b", "doc-c"}
    assert batch.ranked_desc[0].source_id == "doc-a"
    assert batch.ranked_desc[0].rerank_score == 0.95
    assert [p.rerank_score for p in batch.ranked_desc] == sorted(
        (p.rerank_score for p in batch.ranked_desc), reverse=True
    )
    assert batch.passages == tuple(reversed(batch.ranked_desc))


async def test_retrieve_scales_top_k_for_five_scoped_papers() -> None:
    doc_ids = tuple(f"doc-{i}" for i in range(5))
    hits = [
        KnowledgeHit(
            source_id=doc_id,
            source_type="paper",
            title=f"Paper {doc_id}",
            content=f"{doc_id} chunk {chunk_idx}",
            score=0.90 - idx * 0.02 - chunk_idx * 0.01,
            chunk_index=chunk_idx,
        )
        for idx, doc_id in enumerate(doc_ids)
        for chunk_idx in range(3)
    ]
    score_map = {(h.source_id, h.chunk_index): h.score for h in hits}
    engine = RetrievalEngine(
        knowledge_store=_FakeKnowledgeStore(hits),
        reranker=_FixedScoreReranker(score_map),
        default_top_k=6,
        default_min_score=0.05,
    )

    batch = await engine.retrieve(
        RetrievalQuery(
            space_id="sp-1",
            primary_query="synthesize findings across 5 papers",
            source_ids=doc_ids,
        )
    )

    assert len(batch.ranked_desc) == 10
    assert {p.source_id for p in batch.ranked_desc} == set(doc_ids)


async def test_retrieve_preserves_base_top_k_for_single_and_unscoped_queries() -> None:
    hits = [
        KnowledgeHit(
            source_id="doc-a",
            source_type="paper",
            title="Paper A",
            content=f"Chunk {idx}",
            score=0.95 - idx * 0.05,
            chunk_index=idx,
        )
        for idx in range(10)
    ]
    score_map = {(h.source_id, h.chunk_index): h.score for h in hits}
    engine = RetrievalEngine(
        knowledge_store=_FakeKnowledgeStore(hits),
        reranker=_FixedScoreReranker(score_map),
        default_top_k=6,
        default_min_score=0.05,
    )

    single_batch = await engine.retrieve(
        RetrievalQuery(
            space_id="sp-1",
            primary_query="single paper details",
            source_ids=("doc-a",),
        )
    )
    unscoped_batch = await engine.retrieve(
        RetrievalQuery(
            space_id="sp-1",
            primary_query="unscoped details",
        )
    )

    assert len(single_batch.ranked_desc) == 6
    assert len(unscoped_batch.ranked_desc) == 6
