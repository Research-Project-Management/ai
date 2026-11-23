"""On-demand Cache-Aside PDF fetcher and indexer for attached Library papers."""

from __future__ import annotations

import asyncio
import logging
import re
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable
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
                if int(raw_len) > self._max_bytes:
                    raise ValueError(f"File size {raw_len} exceeds max_bytes={self._max_bytes}")
            except ValueError as exc:
                if "max_bytes=" in str(exc):
                    raise

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


class LazyPdfIndexer:
    """Ensures attached Library PDFs are indexed into Qdrant before RAG retrieval."""

    def __init__(
        self,
        *,
        ingestion_pipeline: IngestionPipeline,
        knowledge_store: KnowledgeStore,
        file_fetcher: FileContentFetcher,
        max_per_turn: int = 3,
        per_file_timeout_seconds: float = 45.0,
    ) -> None:
        self._ingestion_pipeline = ingestion_pipeline
        self._knowledge_store = knowledge_store
        self._file_fetcher = file_fetcher
        self._max_per_turn = max(1, max_per_turn)
        self._per_file_timeout_seconds = per_file_timeout_seconds
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

    async def ensure_indexed(
        self,
        *,
        space_id: str,
        sources: Sequence[AttachedSource],
    ) -> LazyIndexReport:
        """Index missing PDF-backed sources lazily while skipping metadata-only items."""
        ready: list[str] = []
        ready_seen: set[str] = set()
        newly_indexed: list[str] = []
        skipped_metadata: list[str] = []
        failed: list[dict[str, str]] = []
        indexed_count = 0

        def _mark_ready(sid: str) -> None:
            if sid not in ready_seen:
                ready_seen.add(sid)
                ready.append(sid)

        for src in sources:
            if src.kind == "upload":
                sid = src.effective_source_id
                if sid:
                    _mark_ready(sid)
                continue

            sid = src.effective_source_id
            if not sid:
                label = src.item_id or src.title or "unknown"
                skipped_metadata.append(label)
                continue

            try:
                if await self._knowledge_store.has_source(space_id=space_id, source_id=sid):
                    _mark_ready(sid)
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("has_source check failed for %s: %s", sid, exc)

            if not src.file_id:
                failed.append(
                    {
                        "source_id": sid,
                        "title": src.title,
                        "error": "not indexed and no file_id",
                    }
                )
                continue

            if indexed_count >= self._max_per_turn:
                failed.append(
                    {
                        "source_id": sid,
                        "title": src.title,
                        "error": "lazy_index_turn_limit_reached",
                    }
                )
                continue

            lock = self._lock_for(space_id, sid)
            try:
                async with lock:
                    if await self._knowledge_store.has_source(space_id=space_id, source_id=sid):
                        _mark_ready(sid)
                        continue

                    raw_bytes, filename, mime_type = await asyncio.wait_for(
                        self._file_fetcher.fetch_bytes(src.file_id),
                        timeout=self._per_file_timeout_seconds,
                    )
                    meta: dict[str, str] = {"file_id": src.file_id}
                    if src.item_id:
                        meta["item_id"] = src.item_id

                    await asyncio.wait_for(
                        self._ingestion_pipeline.ingest(
                            IngestDocumentRequest(
                                space_id=space_id,
                                source_type="file",
                                source_id=sid,
                                title=src.title or filename,
                                filename=filename,
                                mime_type=mime_type,
                                raw_bytes=raw_bytes,
                                metadata=meta,
                            )
                        ),
                        timeout=self._per_file_timeout_seconds,
                    )
                    indexed_count += 1
                    _mark_ready(sid)
                    if sid not in newly_indexed:
                        newly_indexed.append(sid)
            except Exception as exc:  # noqa: BLE001 - keep turn alive on individual file errors
                logger.warning("Lazy PDF index failed for source_id=%s: %s", sid, exc)
                failed.append(
                    {
                        "source_id": sid,
                        "title": src.title,
                        "error": str(exc)[:200],
                    }
                )

        return LazyIndexReport(
            ready_source_ids=tuple(ready),
            newly_indexed=tuple(newly_indexed),
            skipped_metadata_only=tuple(skipped_metadata),
            failed=tuple(failed),
        )
