"""On-demand Cache-Aside PDF fetcher and indexer for attached Library papers."""

from __future__ import annotations

import asyncio
import logging
import re
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable
from urllib.parse import quote, unquote, urljoin, urlparse

import httpx

from ai.agents.rag.ingestion.contracts import IngestDocumentRequest
from ai.agents.rag.ingestion.pipeline import IngestionPipeline
from ai.shared.contracts.auth import get_delegation_token
from ai.shared.contracts.knowledge import KnowledgeStore
from ai.shared.contracts.sources import AttachedSource

logger = logging.getLogger(__name__)

_REDIRECT_STATUSES: frozenset[int] = frozenset({301, 302, 303, 307, 308})
_FILENAME_STAR_RE = re.compile(r"filename\*\s*=\s*(?:UTF-8''|utf-8'')?([^;\s]+)", re.IGNORECASE)
_FILENAME_RE = re.compile(r'filename\s*=\s*"([^"]+)"|filename\s*=\s*([^;\s]+)', re.IGNORECASE)


@runtime_checkable
class FileContentFetcher(Protocol):
    """Port interface for downloading raw attached file bytes by Backend file ID."""

    async def fetch_bytes(self, file_id: str) -> tuple[bytes, str, str]:
        """Return `(raw_bytes, filename, mime_type)` for `file_id`."""
        ...


def _extract_filename(content_disposition: str | None, *, fallback: str) -> str:
    if not content_disposition:
        return fallback
    star_match = _FILENAME_STAR_RE.search(content_disposition)
    if star_match:
        decoded = Path(unquote(star_match.group(1)).strip().strip('"')).name
        if decoded:
            return decoded
    basic_match = _FILENAME_RE.search(content_disposition)
    if basic_match:
        candidate = basic_match.group(1) or basic_match.group(2) or ""
        cleaned = Path(candidate.strip().strip('"')).name
        if cleaned:
            return cleaned
    return fallback


class HttpFileContentFetcher:
    """Downloads file bytes from Backend `/api/v1/files/{id}/content` with safe redirects."""

    def __init__(
        self,
        *,
        backend_url: str,
        timeout_seconds: float = 20.0,
        max_bytes: int = 50 * 1024 * 1024,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._backend_url = backend_url.strip().rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._max_bytes = max_bytes
        self._http_client = http_client

    async def fetch_bytes(self, file_id: str) -> tuple[bytes, str, str]:
        """Download binary bytes using active delegation token; strip auth on cross-origin redirects."""
        clean_id = file_id.strip()
        if not clean_id:
            raise ValueError("file_id must not be empty")
        if not self._backend_url:
            raise RuntimeError("LIBRARY_BACKEND_URL is not configured for file download")

        token = get_delegation_token()
        if not token:
            raise RuntimeError("Missing user delegation token for file download")

        if self._http_client is not None:
            return await self._download(self._http_client, clean_id, token)

        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            return await self._download(client, clean_id, token)

    async def _download(
        self,
        client: httpx.AsyncClient,
        file_id: str,
        token: str,
    ) -> tuple[bytes, str, str]:
        base_parsed = urlparse(self._backend_url)
        url = f"{self._backend_url}/api/v1/files/{quote(file_id, safe='')}/content"
        headers: dict[str, str] = {"Authorization": f"Bearer {token}"}

        resp: httpx.Response | None = None
        for _ in range(4):
            resp = await client.get(url, headers=headers, follow_redirects=False)
            if resp.status_code not in _REDIRECT_STATUSES:
                break
            location = (resp.headers.get("location") or "").strip()
            if not location:
                raise RuntimeError(f"Redirect without Location header for file {file_id}")
            next_url = urljoin(url, location)
            next_parsed = urlparse(next_url)
            same_origin = (next_parsed.scheme, next_parsed.netloc) == (
                base_parsed.scheme,
                base_parsed.netloc,
            )
            # Never forward Flux Bearer tokens to external presigned S3/R2 URLs.
            headers = {"Authorization": f"Bearer {token}"} if same_origin else {}
            url = next_url

        if resp is None:
            raise RuntimeError(f"No HTTP response received for file {file_id}")
        if resp.status_code in _REDIRECT_STATUSES:
            raise RuntimeError(f"Too many redirects downloading file {file_id}")
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to download file {file_id}: HTTP {resp.status_code}")

        raw_len = resp.headers.get("content-length")
        if raw_len:
            try:
                declared_len = int(raw_len)
            except ValueError:
                declared_len = None
            if declared_len is not None and declared_len > self._max_bytes:
                raise ValueError(f"File size {raw_len} exceeds max_bytes={self._max_bytes}")

        raw_bytes = resp.content
        if not raw_bytes:
            raise ValueError(f"Downloaded file {file_id} is empty")
        if len(raw_bytes) > self._max_bytes:
            raise ValueError(f"File size {len(raw_bytes)} exceeds max_bytes={self._max_bytes}")

        filename = _extract_filename(
            resp.headers.get("content-disposition"),
            fallback=f"{file_id}.pdf",
        )
        mime_raw = resp.headers.get("content-type") or "application/pdf"
        mime_type = mime_raw.split(";", 1)[0].strip().lower() or "application/pdf"
        return raw_bytes, filename, mime_type


@dataclass(frozen=True)
class LazyIndexReport:
    """Outcome summary of Cache-Aside vector availability checks for attached sources."""

    ready_source_ids: tuple[str, ...] = ()
    newly_indexed: tuple[str, ...] = ()
    skipped_metadata_only: tuple[str, ...] = ()
    failed: tuple[dict[str, str], ...] = ()

    def resolve_source_ids(
        self,
        attached: Sequence[AttachedSource],
        raw_doc_ids: Sequence[str],
    ) -> tuple[str, ...]:
        """Combine ready source IDs with non-failed, non-metadata-only fallback IDs."""
        metadata_only_items = {
            s.item_id for s in attached if s.item_id and not s.effective_source_id
        }
        failed_ids = {f["source_id"] for f in self.failed if f.get("source_id")}
        extra_fallbacks = tuple(
            doc_id
            for doc_id in raw_doc_ids
            if doc_id not in metadata_only_items
            and doc_id not in failed_ids
            and doc_id not in self.ready_source_ids
        )
        return (*self.ready_source_ids, *extra_fallbacks)

    def to_artifacts(self) -> dict[str, Any]:
        """Build artifact telemetry when lazy indexing took action or had skips/failures."""
        if not (self.skipped_metadata_only or self.failed or self.newly_indexed):
            return {}
        return {
            "lazy_indexed_count": len(self.newly_indexed),
            "skipped_metadata_only": list(self.skipped_metadata_only),
            "lazy_index_failed": list(self.failed),
        }


@dataclass(frozen=True)
class _FileIndexOutcome:
    """Typed per-source outcome produced during lazy PDF indexing."""

    status: Literal["ready", "indexed", "failed"]
    source_id: str
    title: str = ""
    error: str | None = None

    def to_failure_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "error": self.error or "",
        }


class LazyPdfIndexer:
    """Ensures attached Library PDFs are indexed into Qdrant before RAG retrieval."""

    def __init__(
        self,
        *,
        ingestion_pipeline: IngestionPipeline,
        knowledge_store: KnowledgeStore,
        file_fetcher: FileContentFetcher,
        max_per_turn: int = 8,
        per_file_timeout_seconds: float = 45.0,
        max_concurrency: int = 3,
    ) -> None:
        self._ingestion_pipeline = ingestion_pipeline
        self._knowledge_store = knowledge_store
        self._file_fetcher = file_fetcher
        self._max_per_turn = max(1, max_per_turn)
        self._per_file_timeout_seconds = per_file_timeout_seconds
        self._max_concurrency = max(1, max_concurrency)
        self._locks: OrderedDict[tuple[str, str], asyncio.Lock] = OrderedDict()

    def _lock_for(self, space_id: str, source_id: str) -> asyncio.Lock:
        key = (space_id, source_id)
        lock = self._locks.get(key)
        if lock is not None:
            self._locks.move_to_end(key)
            return lock
        lock = asyncio.Lock()
        self._locks[key] = lock
        while len(self._locks) > 512:
            oldest_key, oldest_lock = next(iter(self._locks.items()))
            if not oldest_lock.locked():
                del self._locks[oldest_key]
            else:
                break
        return lock

    async def _index_one(
        self,
        *,
        space_id: str,
        order_idx: int,
        source_id: str,
        file_id: str,
        source: AttachedSource,
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, _FileIndexOutcome]:
        lock = self._lock_for(space_id, source_id)
        try:
            async with lock:
                if await self._knowledge_store.has_source(space_id=space_id, source_id=source_id):
                    return order_idx, _FileIndexOutcome(status="ready", source_id=source_id)

                async with semaphore:
                    raw_bytes, filename, mime_type = await asyncio.wait_for(
                        self._file_fetcher.fetch_bytes(file_id),
                        timeout=self._per_file_timeout_seconds,
                    )
                    meta: dict[str, str] = {"file_id": file_id}
                    if source.item_id:
                        meta["item_id"] = source.item_id

                    await asyncio.wait_for(
                        self._ingestion_pipeline.ingest(
                            IngestDocumentRequest(
                                space_id=space_id,
                                source_type="file",
                                source_id=source_id,
                                title=source.title or filename,
                                filename=filename,
                                mime_type=mime_type,
                                raw_bytes=raw_bytes,
                                metadata=meta,
                            )
                        ),
                        timeout=self._per_file_timeout_seconds,
                    )
                    return order_idx, _FileIndexOutcome(status="indexed", source_id=source_id)
        except Exception as exc:  # noqa: BLE001 - keep turn alive on individual file errors
            logger.warning("Lazy PDF index failed for source_id=%s: %s", source_id, exc)
            return (
                order_idx,
                _FileIndexOutcome(
                    status="failed",
                    source_id=source_id,
                    title=source.title,
                    error=str(exc)[:200],
                ),
            )

    async def ensure_indexed(
        self,
        *,
        space_id: str,
        sources: Sequence[AttachedSource],
    ) -> LazyIndexReport:
        """Index missing PDF-backed sources lazily with bounded concurrency."""
        skipped_metadata: list[str] = []
        outcomes: dict[int, _FileIndexOutcome] = {}
        pending: list[tuple[int, str, str, AttachedSource]] = []
        pending_sids: set[str] = set()

        for idx, src in enumerate(sources):
            if src.kind == "upload":
                sid = src.effective_source_id
                if sid:
                    outcomes[idx] = _FileIndexOutcome(status="ready", source_id=sid)
                continue

            sid = src.effective_source_id
            if not sid:
                label = src.item_id or src.title or "unknown"
                skipped_metadata.append(label)
                continue

            try:
                if await self._knowledge_store.has_source(space_id=space_id, source_id=sid):
                    outcomes[idx] = _FileIndexOutcome(status="ready", source_id=sid)
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("has_source check failed for %s: %s", sid, exc)

            file_id = src.file_id
            if not file_id:
                outcomes[idx] = _FileIndexOutcome(
                    status="failed",
                    source_id=sid,
                    title=src.title,
                    error="not indexed and no file_id",
                )
                continue

            if sid not in pending_sids:
                pending_sids.add(sid)
                pending.append((idx, sid, file_id, src))

        semaphore = asyncio.Semaphore(self._max_concurrency)

        indexed_count = 0
        pos = 0
        while pos < len(pending) and indexed_count < self._max_per_turn:
            budget = self._max_per_turn - indexed_count
            wave = pending[pos : pos + budget]
            pos += len(wave)
            wave_results = await asyncio.gather(
                *(
                    self._index_one(
                        space_id=space_id,
                        order_idx=order_idx,
                        source_id=sid,
                        file_id=file_id,
                        source=src,
                        semaphore=semaphore,
                    )
                    for order_idx, sid, file_id, src in wave
                )
            )
            for order_idx, outcome in wave_results:
                outcomes[order_idx] = outcome
                if outcome.status == "indexed":
                    indexed_count += 1

        for order_idx, sid, _file_id, src in pending[pos:]:
            outcomes[order_idx] = _FileIndexOutcome(
                status="failed",
                source_id=sid,
                title=src.title,
                error="lazy_index_turn_limit_reached",
            )

        ready: list[str] = []
        ready_seen: set[str] = set()
        newly_indexed: list[str] = []
        failed: list[dict[str, str]] = []

        for idx in sorted(outcomes):
            outcome = outcomes[idx]
            if outcome.status in {"ready", "indexed"}:
                if outcome.source_id not in ready_seen:
                    ready_seen.add(outcome.source_id)
                    ready.append(outcome.source_id)
                if outcome.status == "indexed" and outcome.source_id not in newly_indexed:
                    newly_indexed.append(outcome.source_id)
            else:
                failed.append(outcome.to_failure_dict())

        return LazyIndexReport(
            ready_source_ids=tuple(ready),
            newly_indexed=tuple(newly_indexed),
            skipped_metadata_only=tuple(skipped_metadata),
            failed=tuple(failed),
        )
