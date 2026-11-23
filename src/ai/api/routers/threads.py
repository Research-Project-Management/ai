"""Thread inspection, cooperative cancellation, and checkpoint lifecycle endpoints."""

from collections.abc import Mapping
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from ai.api.dependencies import (
    get_cancel_registry,
    get_checkpointer,
    get_mutation_journal,
    verify_internal_key,
)
from ai.api.schemas import (
    SpaceThreadsDeleteResponse,
    ThreadCancelResponse,
    ThreadCheckpointsResponse,
    ThreadDeleteResponse,
    ThreadRewindRequest,
    ThreadRewindResponse,
    ThreadStateResponse,
    TurnRevertRequest,
    TurnRevertResponse,
)
from ai.orchestrator.rewind import (
    CheckpointNotFoundError,
    list_thread_checkpoints,
    rewind_thread_state,
)
from ai.shared.contracts.auth import bind_delegation_token
from ai.shared.contracts.run_control import RunCancellationRegistry
from ai.shared.gateway.transport import BackendTransport
from ai.shared.resilience.mutations import (
    MutationRevertError,
    RevertStepResult,
    TurnMutationJournal,
    revert_turn_mutations,
)

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


def _build_revert_transport(raw_request: Request, *, service: str) -> BackendTransport:
    settings = getattr(raw_request.app.state, "settings", None)
    backend_url = settings.project.backend_url if settings is not None else ""
    internal_key = (
        settings.server.internal_api_key.get_secret_value() if settings is not None else ""
    )
    http_client = getattr(raw_request.app.state, "shared_http_client", None)
    return BackendTransport(
        base_url=backend_url,
        service=service,
        error_factory=MutationRevertError,
        http_client=http_client,
        internal_api_key=internal_key,
    )


@router.post("/turns/{trace_id}/revert", response_model=TurnRevertResponse)
async def revert_turn(
    trace_id: str,
    raw_request: Request,
    journal: Annotated[TurnMutationJournal, Depends(get_mutation_journal)],
    space_id: Annotated[str, Query(min_length=1)],
    body: Annotated[TurnRevertRequest | None, Body()] = None,
) -> TurnRevertResponse:
    """Execute LIFO compensating HTTP actions for mutations recorded in `trace_id`."""
    transport = _build_revert_transport(raw_request, service="Revert")
    auth_header = raw_request.headers.get("authorization", "")
    with bind_delegation_token(auth_header):
        results = await revert_turn_mutations(
            journal,
            space_id=space_id,
            trace_id=trace_id,
            transport=transport,
            entity_ids=body.entity_ids if body is not None else None,
        )

    return TurnRevertResponse(
        trace_id=trace_id,
        space_id=space_id,
        reverted_count=sum(1 for item in results if item.status == "reverted"),
        results=results,
    )


@router.get("/threads/{chat_id}/checkpoints", response_model=ThreadCheckpointsResponse)
async def get_thread_checkpoints(
    chat_id: str,
    raw_request: Request,
    checkpointer: Annotated[BaseCheckpointSaver, Depends(get_checkpointer)],
    space_id: Annotated[str, Query(min_length=1)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> ThreadCheckpointsResponse:
    """List recent LangGraph checkpoints for `chat_id` inside `space_id`."""
    saved = await checkpointer.aget_tuple({"configurable": {"thread_id": chat_id}})
    if saved is None:
        return ThreadCheckpointsResponse(chat_id=chat_id, space_id=space_id, checkpoints=[])

    _assert_thread_in_space(saved.metadata, space_id)
    graph: CompiledStateGraph = raw_request.app.state.root_graph
    checkpoints = await list_thread_checkpoints(graph, chat_id=chat_id, limit=limit)
    return ThreadCheckpointsResponse(
        chat_id=chat_id,
        space_id=space_id,
        checkpoints=checkpoints,
    )


@router.post("/threads/{chat_id}/rewind", response_model=ThreadRewindResponse)
async def rewind_thread(
    chat_id: str,
    raw_request: Request,
    body: ThreadRewindRequest,
    checkpointer: Annotated[BaseCheckpointSaver, Depends(get_checkpointer)],
    journal: Annotated[TurnMutationJournal, Depends(get_mutation_journal)],
    space_id: Annotated[str, Query(min_length=1)],
) -> ThreadRewindResponse:
    """Rewind LangGraph conversation head to a past checkpoint and optionally undo turn mutations."""
    saved = await checkpointer.aget_tuple({"configurable": {"thread_id": chat_id}})
    if saved is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found",
        )
    _assert_thread_in_space(saved.metadata, space_id)

    graph: CompiledStateGraph = raw_request.app.state.root_graph
    try:
        new_id, msg_count = await rewind_thread_state(
            graph,
            chat_id=chat_id,
            checkpoint_id=body.checkpoint_id,
            space_id=space_id,
        )
    except CheckpointNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    all_results: list[RevertStepResult] = []
    trace_ids = [tid.strip() for tid in body.revert_trace_ids if tid and tid.strip()]
    if trace_ids:
        transport = _build_revert_transport(raw_request, service="Rewind")
        auth_header = raw_request.headers.get("authorization", "")
        with bind_delegation_token(auth_header):
            for tid in reversed(trace_ids):
                step_results = await revert_turn_mutations(
                    journal,
                    space_id=space_id,
                    trace_id=tid,
                    transport=transport,
                )
                all_results.extend(step_results)

    return ThreadRewindResponse(
        chat_id=chat_id,
        space_id=space_id,
        target_checkpoint_id=body.checkpoint_id,
        new_checkpoint_id=new_id,
        message_count=msg_count,
        reverted_count=sum(1 for item in all_results if item.status == "reverted"),
        revert_results=all_results,
    )
