"""Space-isolated Hybrid RAG ingestion, retrieval, and cleanup endpoints."""

import json
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from ai.agents.rag.ingestion.contracts import IngestDocumentRequest
from ai.agents.rag.ingestion.digest import upsert_timeline_digest
from ai.agents.rag.ingestion.jobs import (
    IngestionJobRecord,
    IngestionJobRunner,
    IngestionQueueFullError,
)
from ai.agents.rag.ingestion.pipeline import IngestionPipeline
from ai.agents.rag.retrieval.contracts import RetrievalQuery
from ai.agents.rag.retrieval.engine import RetrievalEngine
from ai.api.dependencies import (
    enforce_space_rate_limit,
    get_ingestion_job_runner,
    get_ingestion_pipeline,
    get_knowledge_store,
    get_rate_limiter,
    get_retrieval_engine,
    verify_internal_key,
)
from ai.api.schemas import (
    KnowledgeDeleteResponse,
    KnowledgeIngestResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    KnowledgeTimelineDigestRequest,
    KnowledgeTimelineDigestResponse,
    KnowledgeUpsertRequest,
    KnowledgeUpsertResponse,
)
from ai.shared.contracts.knowledge import (
    KnowledgeChunkInput,
    KnowledgeDocumentInput,
    KnowledgeStore,
)
from ai.shared.contracts.rate_limit import RateLimiter

router = APIRouter(tags=["knowledge"], dependencies=[Depends(verify_internal_key)])


def _to_ingest_response(record: IngestionJobRecord) -> KnowledgeIngestResponse:
    return KnowledgeIngestResponse(
        job_id=record.job_id,
        status=record.status,
        space_id=record.space_id,
        source_type=record.source_type,
        source_id=record.source_id,
        title=record.title,
        pages_parsed=record.pages_parsed,
        chunks_indexed=record.chunks_indexed,
        error=record.error,
    )


@router.post(
    "/knowledge/ingest",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=KnowledgeIngestResponse,
)
async def ingest_knowledge_file(
    file: Annotated[UploadFile, File(...)],
    space_id: Annotated[str, Form(min_length=1)],
    source_type: Annotated[str, Form(min_length=1)],
    source_id: Annotated[str, Form(min_length=1)],
    runner: Annotated[IngestionJobRunner, Depends(get_ingestion_job_runner)],
    limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
    title: Annotated[str, Form()] = "",
    url: Annotated[str | None, Form()] = None,
    metadata_json: Annotated[str, Form()] = "{}",
) -> KnowledgeIngestResponse:
    """Validate and enqueue PDF/DOCX/Markdown/TXT uploads for bounded async indexing."""
    await enforce_space_rate_limit(space_id, limiter)
    try:
        parsed_meta = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="metadata_json must be valid JSON",
        ) from exc

    if not isinstance(parsed_meta, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="metadata_json must decode to a JSON object",
        )

    max_bytes = runner.max_file_bytes
    raw_bytes = await file.read(max_bytes + 1)
    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )
    if len(raw_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size exceeds max_file_bytes={max_bytes}",
        )
    request = IngestDocumentRequest(
        space_id=space_id,
        source_type=source_type,
        source_id=source_id,
        title=title,
        filename=file.filename or "",
        mime_type=file.content_type or "",
        raw_bytes=raw_bytes,
        url=url,
        metadata=parsed_meta,
    )
    try:
        record = await runner.enqueue(request)
    except IngestionQueueFullError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        ) from exc

    return _to_ingest_response(record)


@router.get("/knowledge/jobs/{job_id}", response_model=KnowledgeIngestResponse)
async def get_ingestion_job(
    job_id: str,
    runner: Annotated[IngestionJobRunner, Depends(get_ingestion_job_runner)],
) -> KnowledgeIngestResponse:
    """Return the latest status of an asynchronous knowledge ingestion job."""
    record = await runner.get_job(job_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ingestion job '{job_id}' not found",
        )
    return _to_ingest_response(record)


@router.post("/knowledge/upsert", response_model=KnowledgeUpsertResponse)
async def upsert_knowledge(
    payload: KnowledgeUpsertRequest,
    store: Annotated[KnowledgeStore, Depends(get_knowledge_store)],
    pipeline: Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)],
    limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
) -> KnowledgeUpsertResponse:
    """Embed and idempotently index explicit chunks or structure-chunk plain content."""
    await enforce_space_rate_limit(payload.space_id, limiter)
    try:
        if payload.chunks:
            doc = KnowledgeDocumentInput(
                space_id=payload.space_id,
                source_type=payload.source_type,
                source_id=payload.source_id,
                title=payload.title,
                url=payload.url,
                content=payload.content,
                chunks=tuple(
                    KnowledgeChunkInput(
                        chunk_index=c.chunk_index,
                        content=c.content,
                        page=c.page,
                        metadata=c.metadata,
                    )
                    for c in payload.chunks
                ),
                metadata=payload.metadata,
            )
            chunks_indexed = await store.upsert(doc)
        else:
            ingest_res = await pipeline.ingest(
                IngestDocumentRequest(
                    space_id=payload.space_id,
                    source_type=payload.source_type,
                    source_id=payload.source_id,
                    title=payload.title,
                    filename=f"{payload.source_id}.md",
                    content=payload.content,
                    url=payload.url,
                    metadata=payload.metadata,
                )
            )
            chunks_indexed = ingest_res.chunks_indexed
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return KnowledgeUpsertResponse(
        space_id=payload.space_id,
        source_type=payload.source_type,
        source_id=payload.source_id,
        chunks_indexed=chunks_indexed,
    )


@router.post("/knowledge/search", response_model=KnowledgeSearchResponse)
async def search_knowledge(
    payload: KnowledgeSearchRequest,
    retrieval_engine: Annotated[RetrievalEngine, Depends(get_retrieval_engine)],
    limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
) -> KnowledgeSearchResponse:
    """Execute Hybrid RRF search + rerank strictly within `space_id`."""
    await enforce_space_rate_limit(payload.space_id, limiter)
    batch = await retrieval_engine.retrieve(
        RetrievalQuery(
            space_id=payload.space_id,
            primary_query=payload.query,
            source_types=tuple(payload.source_types),
            source_ids=tuple(payload.source_ids),
            top_k=payload.limit,
        )
    )
    return KnowledgeSearchResponse(
        hits=[passage.to_knowledge_hit() for passage in batch.ranked_desc]
    )


@router.delete("/knowledge/{source_type}/{source_id}", response_model=KnowledgeDeleteResponse)
async def delete_knowledge_source(
    source_type: str,
    source_id: str,
    store: Annotated[KnowledgeStore, Depends(get_knowledge_store)],
    space_id: Annotated[str, Query(min_length=1)],
) -> KnowledgeDeleteResponse:
    """Remove all indexed chunks for a specific source inside `space_id`."""
    deleted = await store.delete_source(
        space_id=space_id,
        source_type=source_type,
        source_id=source_id,
    )
    return KnowledgeDeleteResponse(
        space_id=space_id,
        deleted=deleted,
        source_type=source_type,
        source_id=source_id,
    )


@router.post("/knowledge/timeline-digest", response_model=KnowledgeTimelineDigestResponse)
async def create_timeline_digest(
    payload: KnowledgeTimelineDigestRequest,
    store: Annotated[KnowledgeStore, Depends(get_knowledge_store)],
    limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
) -> KnowledgeTimelineDigestResponse:
    """Compile chronological project events into a CoALA episodic timeline digest."""
    await enforce_space_rate_limit(payload.space_id, limiter)
    try:
        res = await upsert_timeline_digest(
            store,
            space_id=payload.space_id,
            digest_source_id=payload.digest_source_id,
            period_label=payload.period_label,
            events=payload.events,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return KnowledgeTimelineDigestResponse(
        space_id=res.space_id,
        digest_source_id=res.digest_source_id,
        period_label=res.period_label,
        events_included=res.events_included,
        chunks_indexed=res.chunks_indexed,
    )


@router.delete("/spaces/{space_id}/knowledge", response_model=KnowledgeDeleteResponse)
async def delete_space_knowledge(
    space_id: str,
    store: Annotated[KnowledgeStore, Depends(get_knowledge_store)],
) -> KnowledgeDeleteResponse:
    """Purge all indexed vector knowledge belonging to a Flux space."""
    deleted = await store.delete_space(space_id=space_id)
    return KnowledgeDeleteResponse(space_id=space_id, deleted=deleted)
