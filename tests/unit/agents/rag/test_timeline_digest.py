"""Unit tests for ADR-001 Phase 3 Episodic Timeline Digests and timeline retrieval boost."""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient

from ai.agents.rag.ingestion.digest import (
    TimelineEventInput,
    build_timeline_digest_markdown,
    upsert_timeline_digest,
)
from ai.agents.rag.retrieval.contracts import RetrievalQuery
from ai.agents.rag.retrieval.engine import RetrievalEngine, _is_timeline_query
from ai.agents.rag.retrieval.reranker import LexicalHybridReranker
from ai.shared.contracts.knowledge import KnowledgeDocumentInput
from ai.shared.llm.embeddings import DeterministicHashEmbedder
from ai.shared.persistence.knowledge import QdrantKnowledgeStore


def test_build_timeline_digest_markdown_orders_chronologically() -> None:
    events = [
        TimelineEventInput(
            source_id="adr-019",
            source_type="policy",
            title="ADR-019 Concurrency 4",
            effective_at="2025-12-10",
            summary="Raised max_concurrency to 4 with dedicated 4 GB pods.",
            supersedes_source_id="inc-064",
        ),
        TimelineEventInput(
            source_id="m3-bench",
            source_type="report",
            title="Q1 Baseline",
            effective_at="2025-03-28",
            summary="71.4% accuracy and $120/month LLM cost.",
        ),
    ]
    md = build_timeline_digest_markdown("2025 Annual", events)
    assert md.index("2025-03-28") < md.index("2025-12-10")
    assert "(Supersedes: `inc-064`)" in md


def test_is_timeline_query_detects_synthesis_and_progression_queries() -> None:
    assert _is_timeline_query("Tổng hợp xu hướng qua cả 4 mốc Tháng 3 đến Tháng 12")
    assert _is_timeline_query("Compare progression across Q1 and Q4 quarters")
    assert not _is_timeline_query("What is the max_concurrency right now?")


async def test_upsert_timeline_digest_and_boost_on_synthesis_query() -> None:
    qdrant = AsyncQdrantClient(location=":memory:")
    store = QdrantKnowledgeStore(qdrant, DeterministicHashEmbedder(dimensions=64))

    await store.upsert(
        KnowledgeDocumentInput(
            space_id="sp-digest",
            source_type="report",
            source_id="raw-q1",
            title="Q1 Raw Report",
            content="Quarterly accuracy baseline 71.4% and cost $120/month.",
            metadata={"effective_at": "2025-03-28"},
        )
    )
    res = await upsert_timeline_digest(
        store,
        space_id="sp-digest",
        digest_source_id="digest-y1",
        period_label="Months 3-12",
        events=[
            TimelineEventInput(
                source_id="raw-q1",
                source_type="report",
                title="Q1 Raw Report",
                effective_at="2025-03-28",
                summary="Accuracy 71.4%, cost $120/month.",
            ),
            TimelineEventInput(
                source_id="raw-q4",
                source_type="report",
                title="Annual Review",
                effective_at="2025-12-15",
                summary="Accuracy 91.2%, cost $78/month.",
            ),
        ],
    )
    assert res.events_included == 2
    assert res.chunks_indexed >= 1

    engine = RetrievalEngine(
        knowledge_store=store,
        reranker=LexicalHybridReranker(),
        default_min_score=0.0,
    )
    batch = await engine.retrieve(
        RetrievalQuery(
            space_id="sp-digest",
            primary_query="Tổng hợp xu hướng accuracy và cost qua các mốc quý",
            top_k=2,
        )
    )
    await qdrant.close()

    assert batch.ranked_desc
    assert batch.ranked_desc[0].source_type == "timeline_digest"
    assert batch.ranked_desc[0].source_id == "digest-y1"
