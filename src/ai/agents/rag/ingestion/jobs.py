"""Bounded background queue and Redis/in-memory job tracking for RAG document ingestion."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import OrderedDict
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict
from redis.asyncio import Redis

from ai.agents.rag.ingestion.contracts import IngestDocumentRequest
from ai.agents.rag.ingestion.pipeline import IngestionPipeline
from ai.shared.persistence.redis import await_redis

logger = logging.getLogger(__name__)

IngestionJobStatus = Literal["queued", "running", "completed", "failed"]

_JOB_KEY_PREFIX = "ai:ingest:job:"
_MEMORY_JOB_CAPACITY = 1000


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


class IngestionQueueFullError(RuntimeError):
    """Raised when active + queued ingestion tasks reach `max_queue_size`."""


class IngestionJobRecord(BaseModel):
    """Immutable snapshot of a single asynchronous ingestion job."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    status: IngestionJobStatus
    space_id: str
    source_type: str
    source_id: str
    title: str = ""
    pages_parsed: int = 0
    chunks_indexed: int = 0
    error: str = ""
    created_at: str
    updated_at: str


@runtime_checkable
class IngestionJobStore(Protocol):
    """Persistence port for ingestion job status records."""

    async def save(self, record: IngestionJobRecord) -> None:
        """Persist or update `record`."""
        ...

    async def get(self, job_id: str) -> IngestionJobRecord | None:
        """Fetch `record` by `job_id`, or `None` when absent/expired."""
        ...


class InMemoryIngestionJobStore:
    """Bounded in-memory job store for tests and Redis-free local development."""

    def __init__(self, *, capacity: int = _MEMORY_JOB_CAPACITY) -> None:
        self._capacity = capacity
        self._records: OrderedDict[str, IngestionJobRecord] = OrderedDict()

    async def save(self, record: IngestionJobRecord) -> None:
        self._records[record.job_id] = record
        self._records.move_to_end(record.job_id)
        if len(self._records) > self._capacity:
            self._records.popitem(last=False)

    async def get(self, job_id: str) -> IngestionJobRecord | None:
        return self._records.get(job_id)


class RedisIngestionJobStore:
    """Redis-backed job store shared across multi-worker uvicorn processes."""

    def __init__(self, redis_client: Redis, *, ttl_seconds: int = 86400) -> None:
        self._redis = redis_client
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _key(job_id: str) -> str:
        return f"{_JOB_KEY_PREFIX}{job_id}"

    async def save(self, record: IngestionJobRecord) -> None:
        await await_redis(
            self._redis.set(
                self._key(record.job_id),
                record.model_dump_json(),
                ex=self._ttl_seconds,
            )
        )

    async def get(self, job_id: str) -> IngestionJobRecord | None:
        raw = await await_redis(self._redis.get(self._key(job_id)))
        if not isinstance(raw, str) or not raw.strip():
            return None
        try:
            return IngestionJobRecord.model_validate_json(raw)
        except Exception as exc:  # noqa: BLE001 - corrupt cache payload should degrade cleanly
            logger.warning("Failed decoding ingestion job %s from Redis (%s)", job_id, exc)
            return None


def build_ingestion_job_store(
    redis_client: Redis | None,
    *,
    ttl_seconds: int = 86400,
) -> IngestionJobStore:
    """Select Redis job store when configured, otherwise bounded in-memory store."""
    if redis_client is not None:
        return RedisIngestionJobStore(redis_client, ttl_seconds=ttl_seconds)
    return InMemoryIngestionJobStore()


class IngestionJobRunner:
    """Bounded async worker queue running `IngestionPipeline.ingest` off the HTTP path."""

    def __init__(
        self,
        pipeline: IngestionPipeline,
        store: IngestionJobStore,
        *,
        max_concurrency: int = 2,
        max_queue_size: int = 50,
    ) -> None:
        self._pipeline = pipeline
        self._store = store
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._max_queue_size = max(1, max_queue_size)
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def max_file_bytes(self) -> int:
        return self._pipeline.max_file_bytes

    async def _save_update(
        self,
        record: IngestionJobRecord,
        **changes: Any,
    ) -> IngestionJobRecord:
        updated = record.model_copy(update={**changes, "updated_at": _utc_now_iso()})
        await self._store.save(updated)
        return updated

    async def _execute_job(
        self,
        queued: IngestionJobRecord,
        request: IngestDocumentRequest,
    ) -> None:
        active = queued
        async with self._semaphore:
            active = await self._save_update(active, status="running")
            try:
                result = await self._pipeline.ingest(request)
                await self._save_update(
                    active,
                    status="completed",
                    title=result.title,
                    pages_parsed=result.pages_parsed,
                    chunks_indexed=result.chunks_indexed,
                    error="",
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - worker boundary persists error on job record
                logger.warning("Ingestion job %s failed (%s)", queued.job_id, exc)
                await self._save_update(active, status="failed", error=str(exc))

    async def enqueue(self, request: IngestDocumentRequest) -> IngestionJobRecord:
        """Accept an ingestion command into the bounded queue or raise `IngestionQueueFullError`."""
        if len(self._tasks) >= self._max_queue_size:
            raise IngestionQueueFullError(
                f"Ingestion queue full (max_queue_size={self._max_queue_size})"
            )

        now = _utc_now_iso()
        record = IngestionJobRecord(
            job_id=uuid.uuid4().hex,
            status="queued",
            space_id=request.space_id,
            source_type=request.source_type,
            source_id=request.source_id,
            title=request.title,
            created_at=now,
            updated_at=now,
        )
        await self._store.save(record)

        task = asyncio.create_task(self._execute_job(record, request))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return record

    async def get_job(self, job_id: str) -> IngestionJobRecord | None:
        """Return the latest status snapshot for `job_id`, or `None` if unknown."""
        return await self._store.get(job_id)

    def snapshot(self) -> dict[str, int]:
        """Return active background job count and bounded queue capacity."""
        return {
            "active_jobs": len(self._tasks),
            "max_queue_size": self._max_queue_size,
        }

    async def aclose(self) -> None:
        """Cancel and await any pending background ingestion tasks during shutdown."""
        pending = list(self._tasks)
        if not pending:
            return
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
