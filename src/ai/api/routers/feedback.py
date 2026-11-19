"""Run quality feedback ingestion and trace lookup endpoints."""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from ai.api.dependencies import get_feedback_sink, verify_internal_key
from ai.api.schemas import FeedbackCreateRequest, FeedbackListResponse
from ai.shared.contracts.run_control import FeedbackRecord, FeedbackSink

router = APIRouter(tags=["feedback"], dependencies=[Depends(verify_internal_key)])


@router.post(
    "/runs/{trace_id}/feedback",
    response_model=FeedbackRecord,
    status_code=status.HTTP_201_CREATED,
)
async def submit_feedback(
    trace_id: str,
    payload: FeedbackCreateRequest,
    sink: Annotated[FeedbackSink, Depends(get_feedback_sink)],
) -> FeedbackRecord:
    """Record a quality evaluation score and comment for an orchestration run."""
    item = FeedbackRecord(
        trace_id=trace_id,
        space_id=payload.space_id,
        score=payload.score,
        comment=payload.comment,
        category=payload.category,
        created_at=datetime.now(UTC).isoformat(),
    )
    await sink.record(item)
    return item


@router.get("/runs/{trace_id}/feedback", response_model=FeedbackListResponse)
async def list_run_feedback(
    trace_id: str,
    sink: Annotated[FeedbackSink, Depends(get_feedback_sink)],
    space_id: Annotated[str, Query(min_length=1)],
) -> FeedbackListResponse:
    """List feedback entries recorded for `trace_id` inside `space_id`."""
    items = await sink.list_by_trace(trace_id, space_id=space_id)
    return FeedbackListResponse(items=items)
