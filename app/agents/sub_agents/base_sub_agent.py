"""
BaseSubAgent — shared template for all workspace ReAct sub-agents.

Eliminates 500+ lines of copy-paste boilerplate.  Subclasses only need to define:
  - card         : AgentCard           (registered via register_card)
  - tools        : list[BaseTool]       (domain-specific LangChain tools)
  - SYSTEM_PROMPT: str                  (template with {workspace_id}, {user_id}, {project_id})
  - _build_prompt(task) -> str          (format SYSTEM_PROMPT with task context)

The graph is pre-compiled once at class-definition time using a class-level
ContextVar for per-request prompt injection — safe for concurrent requests.
"""

from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod
from contextvars import ContextVar
from typing import AsyncIterator, ClassVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from app.agents.card import AgentCard, AgentTask, AgentResult
from app.agents.utils import _extract_token
from app.config import get_agent_llm
from app.tools._client import set_user_context

logger = logging.getLogger(__name__)


class BaseSubAgent:
    """
    Abstract base for all workspace ReAct sub-agents.

    Each concrete subclass must define:
    - ``card``         — AgentCard (already registered)
    - ``tools``        — list of LangChain tools available to this agent
    - ``_build_prompt(task)`` — returns the formatted system prompt string

    The class auto-creates a pre-compiled ReAct graph at first instantiation
    and uses a per-class ContextVar to inject the per-request system prompt
    without recompiling the graph.
    """

    # Subclasses MUST define these
    card: ClassVar[AgentCard]
    tools: ClassVar[list[BaseTool]]

    # Set by _ensure_graph() on first use — one graph per subclass
    _graph: ClassVar | None = None
    _ctx: ClassVar[ContextVar[dict]] | None = None

    @classmethod
    def _ensure_graph(cls) -> None:
        """Lazily build the ReAct graph and ContextVar for this subclass."""
        if cls._graph is not None:
            return

        # Each subclass gets its own ContextVar (name is class-scoped)
        ctx_name = f"_sub_agent_ctx_{cls.__name__}"
        cls._ctx = ContextVar(ctx_name, default={})

        def _dynamic_prompt(state: dict) -> list:
            ctx = cls._ctx.get()  # type: ignore[union-attr]
            system_prompt = ctx.get("system_prompt", "")
            messages = state.get("messages", [])
            if system_prompt:
                return [SystemMessage(content=system_prompt)] + list(messages)
            return list(messages)

        cls._graph = create_react_agent(
            model=get_agent_llm(),
            tools=cls.tools,
            prompt=_dynamic_prompt,
        )
        logger.debug("[%s] ReAct graph compiled.", cls.__name__)

    @abstractmethod
    def _build_prompt(self, task: AgentTask) -> str:
        """Return the formatted system prompt for this request."""
        ...

    def _prepare(self, task: AgentTask) -> list:
        """Set user context + inject prompt, return message list."""
        self._ensure_graph()
        set_user_context(task.user_id, task.workspace_id, task.project_id)
        self.__class__._ctx.set({"system_prompt": self._build_prompt(task)})  # type: ignore[union-attr]
        return task.messages + [HumanMessage(content=task.request)]

    # ── Public API ──────────────────────────────────────────────────────────

    async def invoke(
        self,
        task: AgentTask,
        *,
        timeout: float = 30.0,
        max_retries: int = 1,
    ) -> AgentResult:
        """Run the sub-agent (non-streaming) with timeout + retry."""
        messages = self._prepare(task)
        agent_name = self.card.name
        class_name = self.__class__.__name__

        for attempt in range(max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    self.__class__._graph.ainvoke({"messages": messages}),  # type: ignore[union-attr]
                    timeout=timeout,
                )
                raw_content = result["messages"][-1].content
                content = _extract_token(raw_content) if not isinstance(raw_content, str) else raw_content
                return AgentResult(content=content, agent_name=agent_name)

            except asyncio.TimeoutError:
                logger.warning("[%s] Timeout on attempt %d/%d", class_name, attempt + 1, max_retries + 1)
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return AgentResult(
                    content=f"{class_name} timed out. Please try a simpler request.",
                    agent_name=agent_name,
                    has_error=True,
                )

            except Exception as exc:
                logger.error(
                    "[%s] Error on attempt %d/%d: %s",
                    class_name, attempt + 1, max_retries + 1, exc,
                    exc_info=True,
                )
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return AgentResult(
                    content=f"{class_name} encountered an error: {exc}",
                    agent_name=agent_name,
                    has_error=True,
                )

        # Should never reach here
        return AgentResult(content="Unknown error", agent_name=agent_name, has_error=True)

    async def stream(self, task: AgentTask) -> AsyncIterator[dict]:
        """Stream events from the sub-agent.

        Yields dicts:
          - {"type": "tool_start", "agent": ..., "tool": ..., "input": ...}
          - {"type": "tool_end",   "agent": ..., "tool": ..., "output": ...}
          - {"type": "token",      "content": ...}
          - {"type": "done",       "agent": ...}
        """
        messages = self._prepare(task)
        agent_name = self.card.name

        async for event in self.__class__._graph.astream_events(  # type: ignore[union-attr]
            {"messages": messages}, version="v2"
        ):
            kind = event.get("event", "")

            if kind == "on_tool_start":
                yield {
                    "type": "tool_start",
                    "agent": agent_name,
                    "tool": event.get("name", ""),
                    "input": event.get("data", {}).get("input", {}),
                }

            elif kind == "on_tool_end":
                output = event.get("data", {}).get("output", "")
                if hasattr(output, "content"):
                    output = output.content
                yield {
                    "type": "tool_end",
                    "agent": agent_name,
                    "tool": event.get("name", ""),
                    "output": output,
                }

            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    token = _extract_token(chunk.content)
                    if token:
                        yield {"type": "token", "content": token}

        yield {"type": "done", "agent": agent_name}
