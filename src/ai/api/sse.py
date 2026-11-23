"""Typed SSE serializer orchestrating LangGraph runs through `SseTranslator`."""

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from ai.api.sse_translator import (
    SseTranslator,
    format_sse,
    serialize_artifacts,
    serialize_citations,
)
from ai.shared.contracts.auth import bind_delegation_token
from ai.shared.contracts.run_control import RunCancellationRegistry
from ai.shared.llm.factory import LLMOverride, bind_llm_override
from ai.shared.resilience.mutations import (
    TurnMutationJournal,
    bind_turn_journal,
    build_revert_artifact,
)
from ai.shared.resilience.observability import TokenBudget, ToolExecutionMetrics
from ai.shared.resilience.tool_runner import bind_tool_metrics

logger = logging.getLogger(__name__)


async def stream_graph_events(
    *,
    graph: CompiledStateGraph,
    graph_input: Any,
    run_config: RunnableConfig,
    specialist_names: frozenset[str],
    trace_id: str,
    chat_id: str,
    cancel_registry: RunCancellationRegistry | None = None,
    token_budget: TokenBudget | None = None,
    llm_override: LLMOverride | None = None,
    space_id: str = "",
    delegation_token: str = "",
    tool_metrics: ToolExecutionMetrics | None = None,
    mutation_journal: TurnMutationJournal | None = None,
) -> AsyncIterator[str]:
    """Stream orchestration lifecycle, cancellation, HITL interrupts, and token usage."""
    started = time.perf_counter()
    yield format_sse("run.started", {"trace_id": trace_id, "chat_id": chat_id})
    translator = SseTranslator(specialist_names=specialist_names)

    with (
        bind_delegation_token(delegation_token),
        bind_llm_override(llm_override),
        bind_tool_metrics(tool_metrics),
        bind_turn_journal(mutation_journal, space_id=space_id, trace_id=trace_id),
    ):
        try:
            async for event in graph.astream_events(graph_input, config=run_config, version="v2"):
                if (
                    cancel_registry is not None
                    and space_id
                    and await cancel_registry.is_cancelled(space_id, chat_id)
                ):
                    await cancel_registry.clear(space_id, chat_id)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    final_usage = translator.usage.with_latency(elapsed_ms)
                    if token_budget is not None and space_id:
                        await token_budget.record(space_id, final_usage)
                    revert_art = build_revert_artifact(
                        mutation_journal, space_id=space_id, trace_id=trace_id
                    )
                    cancel_artifacts = (
                        {"revertible_turn": revert_art} if revert_art is not None else {}
                    )
                    yield format_sse(
                        "run.cancelled",
                        {
                            "trace_id": trace_id,
                            "chat_id": chat_id,
                            **({"artifacts": cancel_artifacts} if cancel_artifacts else {}),
                            "usage": final_usage.model_dump(),
                        },
                    )
                    return

                for frame in translator.translate(event):
                    yield frame

            route = translator.observed_route
            citations = translator.observed_citations
            artifacts = translator.observed_artifacts
            if graph.checkpointer is not None:
                snapshot = await graph.aget_state(run_config)
                if snapshot.interrupts:
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    if token_budget is not None and space_id:
                        await token_budget.record(
                            space_id, translator.usage.with_latency(elapsed_ms)
                        )
                    yield format_sse(
                        "hitl.interrupted",
                        {
                            "trace_id": trace_id,
                            "chat_id": chat_id,
                            "interrupts": [item.value for item in snapshot.interrupts],
                        },
                    )
                    return
                values = snapshot.values or {}
                if isinstance(values, dict):
                    route = str(values.get("route") or route or "direct")
                    citations = serialize_citations(values.get("citations")) or citations
                    artifacts = serialize_artifacts(values.get("artifacts")) or artifacts

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            final_usage = translator.usage.with_latency(elapsed_ms)
            if token_budget is not None and space_id:
                await token_budget.record(space_id, final_usage)
            revert_art = build_revert_artifact(
                mutation_journal, space_id=space_id, trace_id=trace_id
            )
            if revert_art is not None:
                artifacts = {**artifacts, "revertible_turn": revert_art}
            yield format_sse(
                "run.finished",
                {
                    "trace_id": trace_id,
                    "route": route,
                    "citations": citations,
                    **({"artifacts": artifacts} if artifacts else {}),
                    "usage": final_usage.model_dump(),
                },
            )
        except Exception as exc:
            logger.exception("Run %s failed", trace_id)
            yield format_sse("run.error", {"trace_id": trace_id, "message": str(exc)})
