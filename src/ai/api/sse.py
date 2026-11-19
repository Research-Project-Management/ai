"""Typed SSE serializer translating LangGraph v2 events into gateway stream events."""

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.constants import TAG_NOSTREAM
from langgraph.graph.state import CompiledStateGraph

from ai.orchestrator.contracts import RoutingDecision
from ai.shared.contracts.auth import bind_delegation_token
from ai.shared.contracts.run_control import RunCancellationRegistry, RunUsage
from ai.shared.llm.factory import LLMOverride, bind_llm_override
from ai.shared.llm.output import extract_text
from ai.shared.resilience.observability import TokenBudget, ToolExecutionMetrics
from ai.shared.resilience.tool_runner import bind_tool_metrics
from ai.shared.workflow.events import PATCH_PREVIEW, TOOL_COMPLETED, TOOL_STARTED

logger = logging.getLogger(__name__)

_LIVE_STREAM_NODES = frozenset(
    {
        "direct_respond",
        "synthesizer",
        "generate_answer",
        "synthesize_answer",
        "synthesize_response",
    }
)


def format_sse(event: str, data: dict[str, Any]) -> str:
    """Format one typed SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _to_json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def serialize_artifacts(raw_artifacts: Any) -> dict[str, Any]:
    if not isinstance(raw_artifacts, dict) or not raw_artifacts:
        return {}
    return {str(k): _to_json_safe(v) for k, v in raw_artifacts.items()}


def serialize_citations(raw_citations: Any) -> list[dict[str, Any]]:
    if not raw_citations:
        return []
    result: list[dict[str, Any]] = []
    for item in raw_citations:
        if hasattr(item, "model_dump"):
            result.append(item.model_dump())
        elif isinstance(item, dict):
            result.append(item)
    return result


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
) -> AsyncIterator[str]:
    """Stream orchestration lifecycle, cancellation, HITL interrupts, and token usage."""
    started = time.perf_counter()
    yield format_sse("run.started", {"trace_id": trace_id, "chat_id": chat_id})
    text_emitted = False
    usage = RunUsage()
    observed_route = "direct"
    observed_citations: list[dict[str, Any]] = []
    observed_artifacts: dict[str, Any] = {}

    with (
        bind_delegation_token(delegation_token),
        bind_llm_override(llm_override),
        bind_tool_metrics(tool_metrics),
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
                    final_usage = usage.with_latency(elapsed_ms)
                    if token_budget is not None and space_id:
                        await token_budget.record(space_id, final_usage)
                    yield format_sse(
                        "run.cancelled",
                        {
                            "trace_id": trace_id,
                            "chat_id": chat_id,
                            "usage": final_usage.model_dump(),
                        },
                    )
                    return

                kind = event.get("event", "")
                name = event.get("name", "")
                metadata = event.get("metadata") or {}
                node_name = metadata.get("langgraph_node", "")
                tags = event.get("tags") or ()

                if kind == "on_chain_start" and name in specialist_names and node_name == name:
                    yield format_sse("agent.started", {"agent": name})

                elif kind == "on_chain_end":
                    output = (event.get("data") or {}).get("output")
                    if name == "supervisor" and isinstance(output, dict):
                        decision = output.get("routing_decision")
                        if isinstance(output.get("route"), str) and output["route"]:
                            observed_route = output["route"]
                        if isinstance(decision, RoutingDecision):
                            yield format_sse(
                                "agent.routed",
                                {
                                    "mode": decision.mode,
                                    "targets": [t.agent for t in decision.targets],
                                    "needs_followup": decision.needs_followup,
                                    "reasoning": decision.reasoning,
                                },
                            )
                    elif name in specialist_names and node_name == name:
                        yield format_sse("agent.finished", {"agent": name})
                    elif (
                        name == "output_guard"
                        and node_name == "output_guard"
                        and isinstance(output, dict)
                    ):
                        if output.get("citations"):
                            observed_citations = serialize_citations(output.get("citations"))
                        if output.get("artifacts"):
                            observed_artifacts = serialize_artifacts(output.get("artifacts"))
                        final_text = output.get("final_response", "")
                        if not text_emitted and isinstance(final_text, str) and final_text:
                            text_emitted = True
                            yield format_sse("text.delta", {"delta": final_text})

                elif kind == "on_tool_start":
                    yield format_sse("tool.called", {"tool": name, "agent": node_name})

                elif kind == "on_tool_end":
                    yield format_sse("tool.completed", {"tool": name, "agent": node_name})

                elif kind == "on_custom_event":
                    raw_data = event.get("data")
                    if not isinstance(raw_data, dict):
                        continue
                    data: dict[str, Any] = dict(raw_data)
                    if name == TOOL_STARTED:
                        yield format_sse(
                            "tool.called",
                            {
                                "tool": str(data.get("tool") or ""),
                                "agent": str(data.get("agent") or ""),
                            },
                        )
                    elif name == TOOL_COMPLETED:
                        yield format_sse(
                            "tool.completed",
                            {
                                "tool": str(data.get("tool") or ""),
                                "agent": str(data.get("agent") or ""),
                                "status": str(data.get("status") or "ok"),
                            },
                        )
                    elif name == PATCH_PREVIEW:
                        patch_payload = data.get("patch")
                        if patch_payload:
                            yield format_sse(
                                PATCH_PREVIEW,
                                {
                                    "patch": _to_json_safe(patch_payload),
                                    "total_patches": int(data.get("total_patches") or 1),
                                    "agent": str(data.get("agent") or "editor_agent"),
                                },
                            )

                elif kind == "on_chat_model_stream":
                    if TAG_NOSTREAM in tags or node_name not in _LIVE_STREAM_NODES:
                        continue
                    chunk = (event.get("data") or {}).get("chunk")
                    delta = extract_text(getattr(chunk, "content", ""))
                    if delta:
                        text_emitted = True
                        yield format_sse("text.delta", {"delta": delta})

                elif kind == "on_chat_model_end":
                    output_msg = (event.get("data") or {}).get("output")
                    usage = usage.merge(RunUsage.from_message(output_msg))

            route = observed_route
            citations = observed_citations
            artifacts = observed_artifacts
            if graph.checkpointer is not None:
                snapshot = await graph.aget_state(run_config)
                if snapshot.interrupts:
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    if token_budget is not None and space_id:
                        await token_budget.record(space_id, usage.with_latency(elapsed_ms))
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
                    route = str(values.get("route") or observed_route or "direct")
                    citations = serialize_citations(values.get("citations")) or observed_citations
                    artifacts = serialize_artifacts(values.get("artifacts")) or observed_artifacts

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            final_usage = usage.with_latency(elapsed_ms)
            if token_budget is not None and space_id:
                await token_budget.record(space_id, final_usage)
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
