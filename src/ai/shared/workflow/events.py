"""Tool-progress events shared by tool-using agents over LangGraph custom events.

Every agent emits the same two events, carrying its own name in the payload:
`tool.started {tool, agent, input?}` and `tool.completed {tool, agent, status, error?}`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from langchain_core.callbacks import adispatch_custom_event
from langchain_core.runnables import RunnableConfig

TOOL_STARTED = "tool.started"
TOOL_COMPLETED = "tool.completed"
PATCH_PREVIEW = "editor.patch_preview"

_TOOL_INPUT_KEYS = (
    "title",
    "file_path",
    "query",
    "identifier",
    "project_id",
    "work_item_id",
    "item_id",
    "collection_id",
    "action",
    "kind",
    "name",
    "target_bib_file",
)
_MAX_INPUT_VALUE_CHARS = 120


def summarize_tool_input(raw_args: Mapping[str, Any]) -> dict[str, Any]:
    """Extract compact, display-safe tool arguments for live UI progress pills."""
    summary: dict[str, Any] = {}
    for key in _TOOL_INPUT_KEYS:
        val = raw_args.get(key)
        if isinstance(val, str):
            clean = val.strip()
            if clean:
                summary[key] = (
                    clean[:_MAX_INPUT_VALUE_CHARS]
                    if len(clean) <= _MAX_INPUT_VALUE_CHARS
                    else clean[: _MAX_INPUT_VALUE_CHARS - 1] + "…"
                )
        elif isinstance(val, int | float | bool):
            summary[key] = val
    return summary


class ToolEventPublisher(Protocol):
    """Port for publishing live tool telemetry."""

    async def emit_tool_started(
        self,
        tool: str,
        config: RunnableConfig,
        *,
        tool_input: dict[str, Any] | None = None,
    ) -> None: ...

    async def emit_tool_completed(
        self,
        tool: str,
        status: str,
        config: RunnableConfig,
        *,
        error: str | None = None,
    ) -> None: ...


class LangGraphToolEvents:
    """Dispatches tool events as LangGraph custom events for SSE translation."""

    def __init__(self, agent_name: str) -> None:
        self._agent_name = agent_name

    async def dispatch(self, name: str, payload: dict[str, Any], config: RunnableConfig) -> None:
        """Dispatch one custom event, ignoring runs without an active callback manager."""
        try:
            await adispatch_custom_event(
                name, {**payload, "agent": self._agent_name}, config=config
            )
        except RuntimeError:
            pass

    async def emit_tool_started(
        self,
        tool: str,
        config: RunnableConfig,
        *,
        tool_input: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {"tool": tool}
        if tool_input:
            payload["input"] = tool_input
        await self.dispatch(TOOL_STARTED, payload, config)

    async def emit_tool_completed(
        self,
        tool: str,
        status: str,
        config: RunnableConfig,
        *,
        error: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {"tool": tool, "status": status}
        clean_err = error.strip()[:200] if isinstance(error, str) and error.strip() else ""
        if clean_err:
            payload["error"] = clean_err
        await self.dispatch(TOOL_COMPLETED, payload, config)


class NullToolEvents:
    """No-op publisher for isolated non-streaming unit tests."""

    async def emit_tool_started(
        self,
        tool: str,
        config: RunnableConfig,
        *,
        tool_input: dict[str, Any] | None = None,
    ) -> None:
        del tool, config, tool_input

    async def emit_tool_completed(
        self,
        tool: str,
        status: str,
        config: RunnableConfig,
        *,
        error: str | None = None,
    ) -> None:
        del tool, status, config, error
