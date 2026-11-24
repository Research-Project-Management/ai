"""Seam tests for bounded-concurrency LazyPdfIndexer.ensure_indexed."""

from __future__ import annotations

import asyncio

import pymupdf
from qdrant_client import AsyncQdrantClient

from ai.agents.rag.ingestion.chunker import MarkdownStructureChunker
from ai.agents.rag.ingestion.lazy_indexer import LazyPdfIndexer
from ai.agents.rag.ingestion.parsers import build_default_parser_registry
from ai.agents.rag.ingestion.pipeline import IngestionPipeline
from ai.shared.contracts.sources import AttachedSource
from ai.shared.llm.embeddings import DeterministicHashEmbedder
from ai.shared.persistence.knowledge import QdrantKnowledgeStore


def _make_pdf_bytes(title: str, text: str) -> bytes:
    doc = pymupdf.open()
    doc.set_metadata({"title": title})
    page = doc.new_page()
    page.insert_text((72, 72), text)
    raw = doc.tobytes()
    doc.close()
    return raw


async def test_lazy_pdf_indexer_bounds_concurrency_to_three_and_preserves_order() -> None:
    qdrant = AsyncQdrantClient(location=":memory:")
    try:
        store = QdrantKnowledgeStore(qdrant, DeterministicHashEmbedder(dimensions=64))
        pipeline = IngestionPipeline(
            parser_registry=build_default_parser_registry(),
            chunker=MarkdownStructureChunker(),
            knowledge_store=store,
        )

        in_flight = 0
        peak_in_flight = 0

        class _DelayedFetcher:
            async def fetch_bytes(self, file_id: str) -> tuple[bytes, str, str]:
                nonlocal in_flight, peak_in_flight
                in_flight += 1
                peak_in_flight = max(peak_in_flight, in_flight)
                try:
                    await asyncio.sleep(0.04)
                    return (
                        _make_pdf_bytes(
                            title=f"Paper {file_id}",
                            text=f"Unique finding for {file_id}.",
                        ),
                        f"{file_id}.pdf",
                        "application/pdf",
                    )
                finally:
                    in_flight -= 1

        indexer = LazyPdfIndexer(
            ingestion_pipeline=pipeline,
            knowledge_store=store,
            file_fetcher=_DelayedFetcher(),
            max_per_turn=6,
        )
        sources = [
            AttachedSource(
                kind="library",
                title=f"Paper {i}",
                item_id=f"item-{i}",
                file_id=f"file-{i}",
                has_pdf=True,
            )
            for i in range(1, 7)
        ]

        report = await indexer.ensure_indexed(space_id="sp-conc", sources=sources)
        expected = tuple(f"file-{i}" for i in range(1, 7))
        assert report.ready_source_ids == expected
        assert report.newly_indexed == expected
        assert report.failed == ()
        # Must run concurrently (>1) while bounded by Semaphore(3)
        assert 2 <= peak_in_flight <= 3
    finally:
        await qdrant.close()


async def test_lazy_pdf_indexer_isolates_timeout_and_errors_under_concurrency() -> None:
    qdrant = AsyncQdrantClient(location=":memory:")
    try:
        store = QdrantKnowledgeStore(qdrant, DeterministicHashEmbedder(dimensions=64))
        pipeline = IngestionPipeline(
            parser_registry=build_default_parser_registry(),
            chunker=MarkdownStructureChunker(),
            knowledge_store=store,
        )

        class _MixedFetcher:
            async def fetch_bytes(self, file_id: str) -> tuple[bytes, str, str]:
                if file_id == "file-slow":
                    await asyncio.sleep(0.5)
                if file_id == "file-err":
                    raise RuntimeError("corrupt stream")
                await asyncio.sleep(0.01)
                return (
                    _make_pdf_bytes(title=file_id, text=f"Body of {file_id}."),
                    f"{file_id}.pdf",
                    "application/pdf",
                )

        indexer = LazyPdfIndexer(
            ingestion_pipeline=pipeline,
            knowledge_store=store,
            file_fetcher=_MixedFetcher(),
            max_per_turn=3,
            per_file_timeout_seconds=0.08,
        )
        sources = [
            AttachedSource(
                kind="library", title="Ok 1", item_id="i1", file_id="file-ok1", has_pdf=True
            ),
            AttachedSource(
                kind="library", title="Slow", item_id="i2", file_id="file-slow", has_pdf=True
            ),
            AttachedSource(
                kind="library", title="Err", item_id="i3", file_id="file-err", has_pdf=True
            ),
            AttachedSource(
                kind="library", title="Ok 2", item_id="i4", file_id="file-ok2", has_pdf=True
            ),
        ]

        report = await indexer.ensure_indexed(space_id="sp-err", sources=sources)
        assert report.ready_source_ids == ("file-ok1", "file-ok2")
        assert report.newly_indexed == ("file-ok1", "file-ok2")
        failed_ids = [item["source_id"] for item in report.failed]
        assert failed_ids == ["file-slow", "file-err"]
    finally:
        await qdrant.close()
