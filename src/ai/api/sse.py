"""Typed SSE serializer orchestrating LangGraph runs through `SseTranslator`."""

import logging
import time
from collections.abc import AsyncIterator
from typing import Any, Final

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from ai.api.dependencies import bind_run_context
from ai.api.sse_translator import (
    SseTranslator,
    format_sse,
    serialize_artifacts,
    serialize_citations,
)
from ai.shared.contracts.run_control import RunCancellationRegistry, RunUsage
from ai.shared.llm.factory import LLMOverride
from ai.shared.resilience.mutations import JournalStore, build_revert_artifact
from ai.shared.resilience.observability import TokenBudget, ToolExecutionMetrics

logger = logging.getLogger(__name__)
CANCEL_CHECK_INTERVAL_SECONDS: Final[float] = 0.3


async def _finalize_run_usage(
    translator: SseTranslator,
    *,
    started: float,
    space_id: str,
    token_budget: TokenBudget | None,
) -> RunUsage:
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    final_usage = translator.usage.with_latency(elapsed_ms)
    if token_budget is not None and space_id:
        await token_budget.record(space_id, final_usage)
    return final_usage


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
    mutation_journal: JournalStore | None = None,
) -> AsyncIterator[str]:
    """Stream orchestration lifecycle, cancellation, HITL interrupts, and token usage."""
    started = time.perf_counter()
    last_cancel_check = 0.0
    yield format_sse("run.started", {"trace_id": trace_id, "chat_id": chat_id})
    translator = SseTranslator(specialist_names=specialist_names)

    with bind_run_context(
        delegation_token=delegation_token,
        llm_override=llm_override,
        tool_metrics=tool_metrics,
        mutation_journal=mutation_journal,
        space_id=space_id,
        trace_id=trace_id,
    ):
        try:
            async for event in graph.astream_events(graph_input, config=run_config, version="v2"):
                if cancel_registry is not None and space_id:
                    now = time.monotonic()
                    is_token_event = event.get("event") == "on_chat_model_stream"
                    if (
                        not is_token_event
                        or (now - last_cancel_check) >= CANCEL_CHECK_INTERVAL_SECONDS
                    ):
                        last_cancel_check = now
                        if await cancel_registry.is_cancelled(space_id, chat_id):
                            await cancel_registry.clear(space_id, chat_id)
                            final_usage = await _finalize_run_usage(
                                translator,
                                started=started,
                                space_id=space_id,
                                token_budget=token_budget,
                            )
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
                    await _finalize_run_usage(
                        translator,
                        started=started,
                        space_id=space_id,
                        token_budget=token_budget,
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

            final_usage = await _finalize_run_usage(
                translator,
                started=started,
                space_id=space_id,
                token_budget=token_budget,
            )
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
