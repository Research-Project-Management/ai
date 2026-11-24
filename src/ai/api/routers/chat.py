"""Chat and HITL resume endpoints (SSE streaming, synchronous, stateful & stateless)."""

import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from ai.api.dependencies import (
    bind_run_context,
    build_space_context,
    enforce_space_rate_limit,
    enforce_space_token_budget,
    get_cancel_registry,
    get_container,
    get_input_guards,
    get_mutation_journal,
    get_rate_limiter,
    get_token_budget,
    rate_limit_headers,
    verify_internal_key,
)
from ai.api.schemas import ChatMessageInput, ChatRequest, ChatResponse, ResumeRequest
from ai.api.sse import serialize_artifacts, serialize_citations, stream_graph_events
from ai.bootstrap import AppContainer
from ai.orchestrator.contracts import InputGuard, InputGuardViolation
from ai.shared.contracts.context import SpaceContext
from ai.shared.contracts.rate_limit import RateLimiter, RateLimitResult
from ai.shared.contracts.run_control import RunCancellationRegistry, RunUsage
from ai.shared.llm.catalog import resolve_catalog_override
from ai.shared.llm.factory import LLMOverride, parse_llm_override
from ai.shared.resilience.mutations import JournalStore, build_revert_artifact
from ai.shared.resilience.observability import TokenBudget

router = APIRouter(tags=["chat"], dependencies=[Depends(verify_internal_key)])


def _to_langchain_messages(items: list[ChatMessageInput]) -> list[BaseMessage]:
    converted: list[BaseMessage] = []
    for item in items:
        if item.role == "user":
            converted.append(HumanMessage(content=item.content))
        elif item.role == "assistant":
            converted.append(AIMessage(content=item.content))
        else:
            converted.append(SystemMessage(content=item.content))
    return converted


def _build_run_config(space: SpaceContext) -> RunnableConfig:
    return {
        "configurable": space.to_configurable(),
        "metadata": {"space_id": space.space_id},
        "recursion_limit": 25,
    }


def _is_stateless_request(payload: ChatRequest) -> bool:
    return payload.stateless or not bool((payload.chat_id or "").strip())


def _select_graph(container: AppContainer, *, stateless: bool) -> CompiledStateGraph:
    if stateless:
        return container.stateless_root_graph
    return container.root_graph


async def _validate_messages(
    messages: list[BaseMessage],
    input_guards: tuple[InputGuard, ...],
    run_config: RunnableConfig,
) -> list[BaseMessage]:
    validated = messages
    try:
        for guard in input_guards:
            validated = await guard.validate_messages(validated, run_config)
    except InputGuardViolation as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "input_guard_rejected", "reason": str(exc)},
        ) from exc
    return validated


def _build_chat_graph_input(
    payload: ChatRequest,
    space: SpaceContext,
    messages: list[BaseMessage],
) -> dict[str, Any]:
    merged_context = dict(payload.context)
    merged_context.pop("llm_override", None)
    if space.page_id is not None and "page_id" not in merged_context:
        merged_context["page_id"] = space.page_id
    if space.document_ids and "document_ids" not in merged_context:
        merged_context["document_ids"] = list(space.document_ids)
    return {
        "messages": messages,
        "intent_hint": payload.intent_hint,
        "context": merged_context,
    }


async def _prepare_request(
    payload: ChatRequest | ResumeRequest,
    raw_request: Request,
    container: AppContainer,
    limiter: RateLimiter,
    token_budget: TokenBudget,
    cancel_registry: RunCancellationRegistry,
    x_trace_id: str | None,
) -> tuple[SpaceContext, RateLimitResult, RunnableConfig, LLMOverride | None]:
    space = build_space_context(payload, raw_request, x_trace_id=x_trace_id)
    limit_result = await enforce_space_rate_limit(space.space_id, limiter)
    byok_override = (
        parse_llm_override(payload.context) if isinstance(payload, ChatRequest) else None
    )
    if byok_override is None:
        await enforce_space_token_budget(space.space_id, token_budget)
    llm_override = byok_override or (
        resolve_catalog_override(payload.context, container.settings.llm)
        if isinstance(payload, ChatRequest)
        else None
    )
    await cancel_registry.clear(space.space_id, space.chat_id)
    return space, limit_result, _build_run_config(space), llm_override


def _effective_token_budget(
    budget: TokenBudget,
    override: LLMOverride | None,
) -> TokenBudget | None:
    return None if (override is not None and override.is_byok) else budget


def _sse_headers(limit_result: RateLimitResult) -> dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        **rate_limit_headers(limit_result),
    }


def _apply_headers(response: Response, headers: dict[str, str]) -> None:
    for name, value in headers.items():
        response.headers[name] = value


def _extract_response_usage(result: dict[str, Any], started: float) -> RunUsage:
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    raw_usage = result.get("usage")
    base = raw_usage if isinstance(raw_usage, RunUsage) else RunUsage()
    return base.with_latency(elapsed_ms)


async def _build_sync_response(
    *,
    graph: CompiledStateGraph,
    result: dict[str, Any],
    run_config: RunnableConfig,
    trace_id: str,
    started: float,
    stateless: bool,
    space_id: str,
    token_budget: TokenBudget | None,
    mutation_journal: JournalStore | None = None,
) -> ChatResponse:
    usage = _extract_response_usage(result, started)
    if token_budget is not None:
        await token_budget.record(space_id, usage)
    artifacts = serialize_artifacts(result.get("artifacts"))
    revert_art = build_revert_artifact(mutation_journal, space_id=space_id, trace_id=trace_id)
    if revert_art is not None:
        artifacts = {**artifacts, "revertible_turn": revert_art}
    if graph.checkpointer is not None:
        snapshot = await graph.aget_state(run_config)
        if snapshot.interrupts:
            return ChatResponse(
                content="",
                route=str(result.get("route") or "direct"),
                status="interrupted",
                citations=serialize_citations(result.get("citations")),
                artifacts=artifacts,
                interrupts=[item.value for item in snapshot.interrupts],
                usage=usage,
                stateless=stateless,
                trace_id=trace_id,
            )

    return ChatResponse(
        content=str(result.get("final_response") or ""),
        route=str(result.get("route") or "direct"),
        status="completed",
        citations=serialize_citations(result.get("citations")),
        artifacts=artifacts,
        usage=usage,
        stateless=stateless,
        trace_id=trace_id,
    )


def _stream_graph_response(
    *,
    container: AppContainer,
    graph: CompiledStateGraph,
    graph_input: Any,
    space: SpaceContext,
    limit_result: RateLimitResult,
    run_config: RunnableConfig,
    cancel_registry: RunCancellationRegistry,
    token_budget: TokenBudget,
    llm_override: LLMOverride | None,
    mutation_journal: JournalStore,
) -> StreamingResponse:
    return StreamingResponse(
        stream_graph_events(
            graph=graph,
            graph_input=graph_input,
            run_config=run_config,
            specialist_names=container.agent_registry.names,
            trace_id=space.trace_id,
            chat_id=space.chat_id,
            cancel_registry=cancel_registry,
            token_budget=_effective_token_budget(token_budget, llm_override),
            llm_override=llm_override,
            space_id=space.space_id,
            delegation_token=space.delegation_token.get_secret_value(),
            tool_metrics=container.tool_metrics,
            mutation_journal=mutation_journal,
        ),
        media_type="text/event-stream",
        headers=_sse_headers(limit_result),
    )


async def _invoke_graph_sync(
    *,
    container: AppContainer,
    response: Response,
    graph: CompiledStateGraph,
    graph_input: Any,
    space: SpaceContext,
    limit_result: RateLimitResult,
    run_config: RunnableConfig,
    token_budget: TokenBudget,
    llm_override: LLMOverride | None,
    mutation_journal: JournalStore,
    started: float,
    stateless: bool,
) -> ChatResponse:
    _apply_headers(response, rate_limit_headers(limit_result))
    with bind_run_context(
        delegation_token=space.delegation_token.get_secret_value(),
        llm_override=llm_override,
        tool_metrics=container.tool_metrics,
        mutation_journal=mutation_journal,
        space_id=space.space_id,
        trace_id=space.trace_id,
    ):
        result = await graph.ainvoke(graph_input, config=run_config)
    return await _build_sync_response(
        graph=graph,
        result=result,
        run_config=run_config,
        trace_id=space.trace_id,
        started=started,
        stateless=stateless,
        space_id=space.space_id,
        token_budget=_effective_token_budget(token_budget, llm_override),
        mutation_journal=mutation_journal,
    )


@router.post("/chat")
async def chat_stream(
    payload: ChatRequest,
    raw_request: Request,
    container: Annotated[AppContainer, Depends(get_container)],
    limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
    token_budget: Annotated[TokenBudget, Depends(get_token_budget)],
    cancel_registry: Annotated[RunCancellationRegistry, Depends(get_cancel_registry)],
    input_guards: Annotated[tuple[InputGuard, ...], Depends(get_input_guards)],
    mutation_journal: Annotated[JournalStore, Depends(get_mutation_journal)],
    x_trace_id: Annotated[str | None, Header(alias="X-Trace-Id")] = None,
) -> StreamingResponse:
    space, limit_result, run_config, llm_override = await _prepare_request(
        payload, raw_request, container, limiter, token_budget, cancel_registry, x_trace_id
    )
    lc_messages = await _validate_messages(
        _to_langchain_messages(payload.messages),
        input_guards,
        run_config,
    )
    graph = _select_graph(container, stateless=_is_stateless_request(payload))
    return _stream_graph_response(
        container=container,
        graph=graph,
        graph_input=_build_chat_graph_input(payload, space, lc_messages),
        space=space,
        limit_result=limit_result,
        run_config=run_config,
        cancel_registry=cancel_registry,
        token_budget=token_budget,
        llm_override=llm_override,
        mutation_journal=mutation_journal,
    )


@router.post("/chat/sync", response_model=ChatResponse)
async def chat_sync(
    payload: ChatRequest,
    raw_request: Request,
    response: Response,
    container: Annotated[AppContainer, Depends(get_container)],
    limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
    token_budget: Annotated[TokenBudget, Depends(get_token_budget)],
    cancel_registry: Annotated[RunCancellationRegistry, Depends(get_cancel_registry)],
    input_guards: Annotated[tuple[InputGuard, ...], Depends(get_input_guards)],
    mutation_journal: Annotated[JournalStore, Depends(get_mutation_journal)],
    x_trace_id: Annotated[str | None, Header(alias="X-Trace-Id")] = None,
) -> ChatResponse:
    started = time.perf_counter()
    space, limit_result, run_config, llm_override = await _prepare_request(
        payload, raw_request, container, limiter, token_budget, cancel_registry, x_trace_id
    )
    lc_messages = await _validate_messages(
        _to_langchain_messages(payload.messages),
        input_guards,
        run_config,
    )
    is_stateless = _is_stateless_request(payload)
    graph = _select_graph(container, stateless=is_stateless)
    return await _invoke_graph_sync(
        container=container,
        response=response,
        graph=graph,
        graph_input=_build_chat_graph_input(payload, space, lc_messages),
        space=space,
        limit_result=limit_result,
        run_config=run_config,
        token_budget=token_budget,
        llm_override=llm_override,
        mutation_journal=mutation_journal,
        started=started,
        stateless=is_stateless,
    )


@router.post("/chat/resume")
async def chat_resume_stream(
    payload: ResumeRequest,
    raw_request: Request,
    container: Annotated[AppContainer, Depends(get_container)],
    limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
    token_budget: Annotated[TokenBudget, Depends(get_token_budget)],
    cancel_registry: Annotated[RunCancellationRegistry, Depends(get_cancel_registry)],
    mutation_journal: Annotated[JournalStore, Depends(get_mutation_journal)],
    x_trace_id: Annotated[str | None, Header(alias="X-Trace-Id")] = None,
) -> StreamingResponse:
    space, limit_result, run_config, llm_override = await _prepare_request(
        payload, raw_request, container, limiter, token_budget, cancel_registry, x_trace_id
    )
    return _stream_graph_response(
        container=container,
        graph=container.root_graph,
        graph_input=Command(resume=payload.decision),
        space=space,
        limit_result=limit_result,
        run_config=run_config,
        cancel_registry=cancel_registry,
        token_budget=token_budget,
        llm_override=llm_override,
        mutation_journal=mutation_journal,
    )


@router.post("/chat/resume/sync", response_model=ChatResponse)
async def chat_resume_sync(
    payload: ResumeRequest,
    raw_request: Request,
    response: Response,
    container: Annotated[AppContainer, Depends(get_container)],
    limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
    token_budget: Annotated[TokenBudget, Depends(get_token_budget)],
    cancel_registry: Annotated[RunCancellationRegistry, Depends(get_cancel_registry)],
    mutation_journal: Annotated[JournalStore, Depends(get_mutation_journal)],
    x_trace_id: Annotated[str | None, Header(alias="X-Trace-Id")] = None,
) -> ChatResponse:
    started = time.perf_counter()
    space, limit_result, run_config, llm_override = await _prepare_request(
        payload, raw_request, container, limiter, token_budget, cancel_registry, x_trace_id
    )
    return await _invoke_graph_sync(
        container=container,
        response=response,
        graph=container.root_graph,
        graph_input=Command(resume=payload.decision),
        space=space,
        limit_result=limit_result,
        run_config=run_config,
        token_budget=token_budget,
        llm_override=llm_override,
        mutation_journal=mutation_journal,
        started=started,
        stateless=False,
    )
