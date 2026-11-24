"""Unit tests for ADR-001 Bi-Temporal superseding and time-aware RAG retrieval."""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient

from ai.agents.rag.retrieval.contracts import RetrievalQuery
from ai.agents.rag.retrieval.engine import RetrievalEngine, _is_current_state_query
from ai.agents.rag.retrieval.reranker import LexicalHybridReranker
from ai.shared.contracts.knowledge import KnowledgeDocumentInput
from ai.shared.llm.embeddings import DeterministicHashEmbedder
from ai.shared.persistence.knowledge import QdrantKnowledgeStore


def test_is_current_state_query_distinguishes_updates_from_timelines() -> None:
    assert _is_current_state_query("Giá trị max_concurrency mới nhất hiện tại là bao nhiêu?")
    assert _is_current_state_query("What is the active current policy as of now?")
    assert not _is_current_state_query("Trình tự thay đổi max_concurrency từ tháng 3 đến tháng 12?")
    assert not _is_current_state_query("Trước khi sự cố INC-064 xảy ra, hệ thống làm gì?")


async def test_bitemporal_superseding_and_current_state_ranking() -> None:
    qdrant = AsyncQdrantClient(location=":memory:")
    store = QdrantKnowledgeStore(qdrant, DeterministicHashEmbedder(dimensions=64))

    await store.upsert(
        KnowledgeDocumentInput(
            space_id="sp-temp",
            source_type="project_doc",
            source_id="m6-spec",
            title="Architecture v2 Month 6",
            content="Large PDF ingestion policy max_concurrency = 2 async queue.",
            metadata={"effective_at": "2025-06-20"},
        )
    )
    await store.upsert(
        KnowledgeDocumentInput(
            space_id="sp-temp",
            source_type="project_doc",
            source_id="m12-ops",
            title="Operations Note #12 Month 12",
            content="Large PDF ingestion Grafana memory check and dead-letter retry note.",
            metadata={"effective_at": "2025-12-12"},
        )
    )
    await store.upsert(
        KnowledgeDocumentInput(
            space_id="sp-temp",
            source_type="policy",
            source_id="m12-adr019",
            title="ADR-019 PDF Concurrency Update",
            content="ADR-019 raises large PDF ingestion max_concurrency = 4 on dedicated pods.",
            metadata={
                "effective_at": "2025-12-10",
                "supersedes_source_ids": ["m6-spec"],
            },
        )
    )

    raw_hits = await store.search(
        space_id="sp-temp", query="PDF ingestion max_concurrency", limit=5
    )
    by_id = {hit.source_id: hit for hit in raw_hits}
    assert by_id["m6-spec"].metadata["lifecycle_status"] == "superseded"
    assert by_id["m6-spec"].metadata["valid_to"] == "2025-12-10"
    assert by_id["m12-adr019"].metadata["lifecycle_status"] == "active"

    engine = RetrievalEngine(
        knowledge_store=store,
        reranker=LexicalHybridReranker(),
        default_min_score=0.0,
    )
    current_batch = await engine.retrieve(
        RetrievalQuery(
            space_id="sp-temp",
            primary_query="Giá trị max_concurrency mới nhất PDF ingestion",
            top_k=3,
        )
    )
    await qdrant.close()

    assert len(current_batch.ranked_desc) >= 2
    assert current_batch.ranked_desc[0].source_id == "m12-adr019"
