"""Tests for bounded async RAG ingestion queue, Redis/memory job store, and API contract."""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import Any, cast

import httpx
from fastapi import FastAPI
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr

from ai.agents.rag.ingestion.chunker import MarkdownStructureChunker
from ai.agents.rag.ingestion.contracts import IngestDocumentRequest
from ai.agents.rag.ingestion.jobs import (
    IngestionJobRecord,
    IngestionJobRunner,
    InMemoryIngestionJobStore,
    RedisIngestionJobStore,
    build_ingestion_job_store,
)
from ai.agents.rag.ingestion.parsers import build_default_parser_registry
from ai.agents.rag.ingestion.pipeline import IngestionPipeline
from ai.main import create_app
from ai.shared.config.database import DatabaseConfig, PostgresConfig, QdrantConfig, RedisConfig
from ai.shared.config.llm import LLMConfig
from ai.shared.config.rag import IngestionConfig
from ai.shared.config.server import AppEnvironment, ServerConfig
from ai.shared.config.settings import Settings
from ai.shared.contracts.knowledge import KnowledgeDocumentInput, KnowledgeHit
from ai.shared.llm.factory import ChatRunnable


def _stub_llm() -> ChatRunnable:
    return RunnableLambda(lambda _messages: AIMessage(content="ok"))


def _test_settings(*, max_concurrency: int = 2, max_queue_size: int = 10) -> Settings:
    return Settings(
        server=ServerConfig(
            app_env=AppEnvironment.TEST,
            internal_api_key=SecretStr("test-key"),
        ),
        llm=LLMConfig(api_key=SecretStr("test-llm-key")),
        ingestion=IngestionConfig(
            max_concurrency=max_concurrency,
            max_queue_size=max_queue_size,
        ),
        database=DatabaseConfig(
            postgres=PostgresConfig(database_url=SecretStr("")),
            redis=RedisConfig(redis_url=SecretStr("")),
            qdrant=QdrantConfig(qdrant_url=""),
        ),
    )


async def _poll_until_settled(
    client: httpx.AsyncClient,
    job_id: str,
    headers: dict[str, str],
    *,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        res = await client.get(f"/knowledge/jobs/{job_id}", headers=headers)
        assert res.status_code == 200
        body = res.json()
        if body["status"] in {"completed", "failed"}:
            return body
        await asyncio.sleep(0.02)
    raise AssertionError(f"Ingestion job {job_id} did not settle within {timeout_s}s")


async def test_ingest_endpoint_enqueues_202_and_polls_to_completion() -> None:
    app: FastAPI = create_app(settings=_test_settings(), llm=_stub_llm())
    headers = {"X-Internal-Key": "test-key"}
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client,
    ):
        missing = await client.get("/knowledge/jobs/nonexistent", headers=headers)
        assert missing.status_code == 404

        empty_upload = await client.post(
            "/knowledge/ingest",
            headers=headers,
            data={"space_id": "sp-1", "source_type": "upload", "source_id": "empty-1"},
            files={"file": ("empty.md", b"", "text/markdown")},
        )
        assert empty_upload.status_code == 400

        res = await client.post(
            "/knowledge/ingest",
            headers=headers,
            data={
                "space_id": "sp-1",
                "source_type": "upload",
                "source_id": "doc-md-1",
                "title": "Design Spec",
            },
            files={
                "file": (
                    "spec.md",
                    b"# Architecture\nBounded ingestion avoids timeouts and protects SSE latency.",
                    "text/markdown",
                )
            },
        )
        assert res.status_code == 202
        queued = res.json()
        assert queued["status"] == "queued"
        assert queued["job_id"]

        settled = await _poll_until_settled(client, queued["job_id"], headers)
        assert settled["status"] == "completed"
        assert settled["title"] == "Design Spec"
        assert settled["pages_parsed"] == 1
        assert settled["chunks_indexed"] >= 1


async def test_ingest_failed_job_records_error() -> None:
    app: FastAPI = create_app(settings=_test_settings(), llm=_stub_llm())
    headers = {"X-Internal-Key": "test-key"}
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client,
    ):
        res = await client.post(
            "/knowledge/ingest",
            headers=headers,
            data={"space_id": "sp-1", "source_type": "upload", "source_id": "bad-1"},
            files={"file": ("malware.exe", b"MZ-binary", "application/octet-stream")},
        )
        assert res.status_code == 202
        job_id = res.json()["job_id"]

        settled = await _poll_until_settled(client, job_id, headers)
        assert settled["status"] == "failed"
        assert settled["error"]


async def test_ingest_queue_full_returns_429() -> None:
    class _StubStore:
        async def upsert(self, doc: KnowledgeDocumentInput) -> int:
            return len(doc.chunks)

        async def search(self, *args: Any, **kwargs: Any) -> list[KnowledgeHit]:
            return []

        async def delete_source(self, *args: Any, **kwargs: Any) -> int:
            return 0

        async def delete_space(self, *args: Any, **kwargs: Any) -> int:
            return 0

    gate = asyncio.Event()
    started = asyncio.Event()

    class _BlockingPipeline(IngestionPipeline):
        async def ingest(self, request: IngestDocumentRequest):  # type: ignore[override]
            started.set()
            await gate.wait()
            return await super().ingest(request)

    app: FastAPI = create_app(
        settings=_test_settings(max_concurrency=1, max_queue_size=1),
        llm=_stub_llm(),
    )
    headers = {"X-Internal-Key": "test-key"}
    blocking_pipeline = _BlockingPipeline(
        parser_registry=build_default_parser_registry(),
        chunker=MarkdownStructureChunker(),
        knowledge_store=cast(Any, _StubStore()),
    )
    runner = IngestionJobRunner(
        blocking_pipeline,
        InMemoryIngestionJobStore(),
        max_concurrency=1,
        max_queue_size=1,
    )

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client,
    ):
        old_container = app.state.container
        app.state.container = replace(old_container, ingestion_job_runner=runner)
        try:
            first = await client.post(
                "/knowledge/ingest",
                headers=headers,
                data={"space_id": "sp-1", "source_type": "upload", "source_id": "doc-1"},
                files={"file": ("doc1.md", b"# One\nFirst queued document", "text/markdown")},
            )
            assert first.status_code == 202
            await asyncio.wait_for(started.wait(), timeout=2.0)

            second = await client.post(
                "/knowledge/ingest",
                headers=headers,
                data={"space_id": "sp-1", "source_type": "upload", "source_id": "doc-2"},
                files={"file": ("doc2.md", b"# Two\nSecond overflow document", "text/markdown")},
            )
            assert second.status_code == 429

            gate.set()
            settled = await _poll_until_settled(client, first.json()["job_id"], headers)
            assert settled["status"] == "completed"
        finally:
            gate.set()
            await runner.aclose()
            app.state.container = old_container


async def test_redis_ingestion_job_store_round_trip() -> None:
    class _FakeRedis:
        def __init__(self) -> None:
            self.kv: dict[str, tuple[str, int | None]] = {}

        async def set(self, key: str, value: str, *, ex: int | None = None) -> bool:
            self.kv[key] = (value, ex)
            return True

        async def get(self, key: str) -> str | None:
            item = self.kv.get(key)
            return item[0] if item is not None else None

    assert isinstance(build_ingestion_job_store(None), InMemoryIngestionJobStore)

    fake_redis = _FakeRedis()
    store = build_ingestion_job_store(cast(Any, fake_redis), ttl_seconds=3600)
    assert isinstance(store, RedisIngestionJobStore)

    rec = IngestionJobRecord(
        job_id="job-123",
        status="completed",
        space_id="sp-redis",
        source_type="file",
        source_id="src-1",
        title="Paper",
        pages_parsed=3,
        chunks_indexed=9,
        created_at="2026-11-23T10:00:00+00:00",
        updated_at="2026-11-23T10:00:02+00:00",
    )
    await store.save(rec)
    assert fake_redis.kv["ai:ingest:job:job-123"][1] == 3600
    loaded = await store.get("job-123")
    assert loaded == rec
    assert await store.get("missing-job") is None

    fake_redis.kv["ai:ingest:job:corrupt"] = ("{not-json", 3600)
    assert await store.get("corrupt") is None
