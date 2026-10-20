"""Tool-progress events shared by every tool-using agent (Observer over LangGraph custom events).

Every agent emits the same two events, carrying its own name in the payload:
`tool.started {tool, agent}` and `tool.completed {tool, agent, status}`.
"""

from __future__ import annotations

from typing import Any, Protocol

from langchain_core.callbacks import adispatch_custom_event
from langchain_core.runnables import RunnableConfig

TOOL_STARTED = "tool.started"
TOOL_COMPLETED = "tool.completed"
PATCH_PREVIEW = "editor.patch_preview"


class ToolEventPublisher(Protocol):
    """Port for publishing live tool telemetry."""

    async def emit_tool_started(self, tool: str, config: RunnableConfig) -> None: ...

    async def emit_tool_completed(
        self, tool: str, status: str, config: RunnableConfig
    ) -> None: ...


class LangGraphToolEvents:
    """Dispatches tool events as LangGraph custom events for SSE translation."""

    def __init__(self, agent_name: str) -> None:
        self._agent_name = agent_name

    async def dispatch(
        self, name: str, payload: dict[str, Any], config: RunnableConfig
    ) -> None:
        """Dispatch one custom event, ignoring runs without an active callback manager."""
        try:
            await adispatch_custom_event(
                name, {**payload, "agent": self._agent_name}, config=config
            )
        except RuntimeError:
            pass

    async def emit_tool_started(self, tool: str, config: RunnableConfig) -> None:
        await self.dispatch(TOOL_STARTED, {"tool": tool}, config)

    async def emit_tool_completed(
        self, tool: str, status: str, config: RunnableConfig
    ) -> None:
        await self.dispatch(TOOL_COMPLETED, {"tool": tool, "status": status}, config)


class NullToolEvents:
    """No-op publisher for isolated non-streaming unit tests."""

    async def emit_tool_started(self, tool: str, config: RunnableConfig) -> None:
        del tool, config

    async def emit_tool_completed(
        self, tool: str, status: str, config: RunnableConfig
    ) -> None:
        del tool, status, config
