"""Strategy-dispatch translator converting LangGraph v2 events into SSE frames."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langgraph.constants import TAG_NOSTREAM

from ai.orchestrator.contracts import AgentName, RoutingDecision
from ai.shared.contracts.run_control import RunUsage
from ai.shared.llm.output import StreamChannelSplitter
from ai.shared.workflow.events import PATCH_PREVIEW, TOOL_COMPLETED, TOOL_STARTED

LIVE_STREAM_NODES = frozenset(
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


def to_json_safe(value: Any) -> Any:
    """Recursively convert Pydantic models and containers into JSON-safe primitives."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_json_safe(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def serialize_artifacts(raw_artifacts: Any) -> dict[str, Any]:
    if not isinstance(raw_artifacts, dict) or not raw_artifacts:
        return {}
    return {str(k): to_json_safe(v) for k, v in raw_artifacts.items()}


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


class SseTranslator:
    """Stateful strategy dispatcher mapping LangGraph v2 events to gateway SSE frames."""

    def __init__(self, *, specialist_names: frozenset[str]) -> None:
        self._specialists = specialist_names
        self._splitter = StreamChannelSplitter()
        self.usage = RunUsage()
        self.observed_route = "direct"
        self.observed_citations: list[dict[str, Any]] = []
        self.observed_artifacts: dict[str, Any] = {}
        self.text_emitted = False
        self._handlers: dict[str, Callable[[Mapping[str, Any]], list[str]]] = {
            "on_chain_start": self._on_chain_start,
            "on_chain_end": self._on_chain_end,
            "on_tool_start": self._on_tool_start,
            "on_tool_end": self._on_tool_end,
            "on_custom_event": self._on_custom_event,
            "on_chat_model_stream": self._on_chat_stream,
            "on_chat_model_end": self._on_chat_end,
        }

    def translate(self, event: Mapping[str, Any]) -> list[str]:
        """Translate a single LangGraph event into zero or more SSE frames."""
        handler = self._handlers.get(str(event.get("event") or ""))
        if handler is None:
            return []
        return handler(event)

    def _on_chain_start(self, event: Mapping[str, Any]) -> list[str]:
        name = str(event.get("name") or "")
        node_name = str((event.get("metadata") or {}).get("langgraph_node") or "")
        if (name in self._specialists or name == "supervisor") and node_name == name:
            return [format_sse("agent.started", {"agent": name})]
        return []

    def _on_chain_end(self, event: Mapping[str, Any]) -> list[str]:
        name = str(event.get("name") or "")
        node_name = str((event.get("metadata") or {}).get("langgraph_node") or "")
        output = (event.get("data") or {}).get("output")

        if name == "supervisor" and isinstance(output, dict):
            route = output.get("route")
            if isinstance(route, str) and route:
                self.observed_route = route
            decision = output.get("routing_decision")
            if isinstance(decision, RoutingDecision):
                return [
                    format_sse(
                        "agent.routed",
                        {
                            "mode": decision.mode,
                            "targets": [t.agent for t in decision.targets],
                            "needs_followup": decision.needs_followup,
                            "reasoning": decision.reasoning,
                        },
                    )
                ]
            return []

        if name in self._specialists and node_name == name:
            return [format_sse("agent.finished", {"agent": name})]

        if name == "output_guard" and node_name == "output_guard" and isinstance(output, dict):
            if output.get("citations"):
                self.observed_citations = serialize_citations(output.get("citations"))
            if output.get("artifacts"):
                self.observed_artifacts = serialize_artifacts(output.get("artifacts"))
            final_text = output.get("final_response", "")
            if not self.text_emitted and isinstance(final_text, str) and final_text:
                self.text_emitted = True
                return [format_sse("text.delta", {"delta": final_text})]
        return []

    def _on_tool_start(self, event: Mapping[str, Any]) -> list[str]:
        name = str(event.get("name") or "")
        node_name = str((event.get("metadata") or {}).get("langgraph_node") or "")
        return [format_sse("tool.called", {"tool": name, "agent": node_name})]

    def _on_tool_end(self, event: Mapping[str, Any]) -> list[str]:
        name = str(event.get("name") or "")
        node_name = str((event.get("metadata") or {}).get("langgraph_node") or "")
        return [format_sse("tool.completed", {"tool": name, "agent": node_name})]

    def _on_custom_event(self, event: Mapping[str, Any]) -> list[str]:
        name = str(event.get("name") or "")
        raw_data = event.get("data")
        if not isinstance(raw_data, dict):
            return []
        data: dict[str, Any] = dict(raw_data)

        if name == TOOL_STARTED:
            raw_input = data.get("input")
            payload: dict[str, Any] = {
                "tool": str(data.get("tool") or ""),
                "agent": str(data.get("agent") or ""),
            }
            if isinstance(raw_input, dict) and raw_input:
                payload["input"] = to_json_safe(raw_input)
            return [format_sse("tool.called", payload)]

        if name == TOOL_COMPLETED:
            err_val = data.get("error")
            done_payload: dict[str, Any] = {
                "tool": str(data.get("tool") or ""),
                "agent": str(data.get("agent") or ""),
                "status": str(data.get("status") or "ok"),
            }
            if isinstance(err_val, str) and err_val.strip():
                done_payload["error"] = err_val.strip()[:200]
            return [format_sse("tool.completed", done_payload)]

        if name == PATCH_PREVIEW:
            patch_payload = data.get("patch")
            if patch_payload:
                return [
                    format_sse(
                        PATCH_PREVIEW,
                        {
                            "patch": to_json_safe(patch_payload),
                            "total_patches": int(data.get("total_patches") or 1),
                            "agent": str(data.get("agent") or AgentName.EDITOR),
                        },
                    )
                ]
        return []

    def _emit_channel_deltas(self, items: Sequence[tuple[str, str]]) -> list[str]:
        frames: list[str] = []
        for channel, delta in items:
            if channel == "reasoning":
                frames.append(format_sse("reasoning.delta", {"delta": delta}))
            elif delta:
                self.text_emitted = True
                frames.append(format_sse("text.delta", {"delta": delta}))
        return frames

    def _on_chat_stream(self, event: Mapping[str, Any]) -> list[str]:
        tags = event.get("tags") or ()
        node_name = str((event.get("metadata") or {}).get("langgraph_node") or "")
        if TAG_NOSTREAM in tags or node_name not in LIVE_STREAM_NODES:
            return []
        chunk = (event.get("data") or {}).get("chunk")
        return self._emit_channel_deltas(self._splitter.feed_chunk(chunk))

    def _on_chat_end(self, event: Mapping[str, Any]) -> list[str]:
        tags = event.get("tags") or ()
        node_name = str((event.get("metadata") or {}).get("langgraph_node") or "")
        frames: list[str] = []
        if TAG_NOSTREAM not in tags and node_name in LIVE_STREAM_NODES:
            frames.extend(self._emit_channel_deltas(self._splitter.flush()))
        output_msg = (event.get("data") or {}).get("output")
        self.usage = self.usage.merge(RunUsage.from_message(output_msg))
        return frames


__all__ = [
    "LIVE_STREAM_NODES",
    "SseTranslator",
    "format_sse",
    "serialize_artifacts",
    "serialize_citations",
    "to_json_safe",
]
