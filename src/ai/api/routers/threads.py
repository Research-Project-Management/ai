"""Thread inspection, cooperative cancellation, and checkpoint lifecycle endpoints."""

from collections.abc import Mapping
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from ai.api.dependencies import get_cancel_registry, get_checkpointer, verify_internal_key
from ai.api.schemas import (
    SpaceThreadsDeleteResponse,
    ThreadCancelResponse,
    ThreadDeleteResponse,
    ThreadStateResponse,
)
from ai.shared.contracts.run_control import RunCancellationRegistry

router = APIRouter(tags=["threads"], dependencies=[Depends(verify_internal_key)])


def _assert_thread_in_space(metadata: Mapping[str, Any] | None, space_id: str) -> None:
    owner_space = (metadata or {}).get("space_id")
    if owner_space != space_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found in space",
        )


@router.get("/threads/{chat_id}/state", response_model=ThreadStateResponse)
async def get_thread_state(
    chat_id: str,
    raw_request: Request,
    checkpointer: Annotated[BaseCheckpointSaver, Depends(get_checkpointer)],
    space_id: Annotated[str, Query(min_length=1)],
) -> ThreadStateResponse:
    """Inspect current checkpointed state and pending HITL interrupts for a thread."""
    run_config: RunnableConfig = {"configurable": {"thread_id": chat_id}}
    saved = await checkpointer.aget_tuple(run_config)
    if saved is None:
        return ThreadStateResponse(
            chat_id=chat_id,
            space_id=space_id,
            exists=False,
            status="empty",
        )

    _assert_thread_in_space(saved.metadata, space_id)
    graph: CompiledStateGraph = raw_request.app.state.root_graph
    snapshot = await graph.aget_state(run_config)
    values = snapshot.values if isinstance(snapshot.values, dict) else {}
    messages = values.get("messages") or []
    interrupts = [item.value for item in snapshot.interrupts] if snapshot.interrupts else []

    return ThreadStateResponse(
        chat_id=chat_id,
        space_id=space_id,
        exists=True,
        status="interrupted" if interrupts else "idle",
        message_count=len(messages) if isinstance(messages, list) else 0,
        route=str(values.get("route")) if values.get("route") else None,
        interrupts=interrupts,
    )


@router.post("/threads/{chat_id}/cancel", response_model=ThreadCancelResponse)
async def cancel_thread_run(
    chat_id: str,
    cancel_registry: Annotated[RunCancellationRegistry, Depends(get_cancel_registry)],
    space_id: Annotated[str, Query(min_length=1)],
) -> ThreadCancelResponse:
    """Signal cooperative cancellation for any active streaming run on `(space_id, chat_id)`."""
    await cancel_registry.request_cancel(space_id, chat_id)
    return ThreadCancelResponse(chat_id=chat_id, space_id=space_id, status="cancelling")


@router.delete("/threads/{chat_id}", response_model=ThreadDeleteResponse)
async def delete_thread(
    chat_id: str,
    checkpointer: Annotated[BaseCheckpointSaver, Depends(get_checkpointer)],
    space_id: Annotated[str, Query(min_length=1)],
) -> ThreadDeleteResponse:
    """Delete checkpoints belonging to `chat_id` inside `space_id`."""
    saved = await checkpointer.aget_tuple({"configurable": {"thread_id": chat_id}})
    if saved is None:
        return ThreadDeleteResponse(chat_id=chat_id, space_id=space_id, deleted=False)

    _assert_thread_in_space(saved.metadata, space_id)
    await checkpointer.adelete_thread(chat_id)
    return ThreadDeleteResponse(chat_id=chat_id, space_id=space_id, deleted=True)


@router.delete("/spaces/{space_id}/threads", response_model=SpaceThreadsDeleteResponse)
async def delete_space_threads(
    space_id: str,
    checkpointer: Annotated[BaseCheckpointSaver, Depends(get_checkpointer)],
) -> SpaceThreadsDeleteResponse:
    """Purge all persisted checkpoints belonging to a Flux space."""
    thread_ids: set[str] = set()
    async for checkpoint_tuple in checkpointer.alist(None, filter={"space_id": space_id}):
        configurable = checkpoint_tuple.config.get("configurable") or {}
        tid = configurable.get("thread_id")
        if isinstance(tid, str) and tid:
            thread_ids.add(tid)

    for tid in thread_ids:
        await checkpointer.adelete_thread(tid)

    return SpaceThreadsDeleteResponse(space_id=space_id, deleted_threads=len(thread_ids))
