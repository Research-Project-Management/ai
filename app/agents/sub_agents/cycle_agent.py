"""
Cycle Sub-Agent — Handles research sprint / cycle operations.

Invoked by Orchestrator when the user wants to:
- List, create, update, delete research cycles
- Get cycle details and progress
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from app.agents.card import AgentCard, AgentTask, AgentResult, register_card
from app.config import get_agent_llm
from app.tools._client import set_user_context
from app.tools.cycle_tools import (
    list_cycles,
    get_cycle_details,
    create_cycle,
    update_cycle,
    delete_cycle,
)

logger = logging.getLogger(__name__)

CYCLE_AGENT_CARD = register_card(AgentCard(
    name="cycle_agent",
    description=(
        "Manages research cycles (sprints): listing cycles, creating new research sprints, "
        "tracking cycle progress, updating cycle status, and deleting cycles."
    ),
    skills=["list_cycles", "get_cycle_details", "create_cycle", "update_cycle", "delete_cycle"],
    has_destructive_ops=True,
))

CYCLE_TOOLS = [list_cycles, get_cycle_details, create_cycle, update_cycle, delete_cycle]

CYCLE_SYSTEM_PROMPT = """\
You are the **Cycle Agent** — a specialized AI for managing research cycles (sprints) \
in the user's workspace.

## Context
- Workspace ID: {workspace_id}
- Active Project ID: {project_id}
- User ID: {user_id}

## Rules
1. Cycles are research sprints with a start date, end date, and phase (planning/active/completed).
2. Always use ISO format for dates (YYYY-MM-DD).
3. For cycle progress, call get_cycle_details.
4. Respond in the same language as the user.
"""


class CycleSubAgent:
    card = CYCLE_AGENT_CARD

    def _build_prompt(self, task: AgentTask) -> str:
        return CYCLE_SYSTEM_PROMPT.format(
            workspace_id=task.workspace_id or "N/A",
            project_id=task.project_id or "none",
            user_id=task.user_id or "unknown",
        )

    def _build_agent(self, prompt: str):
        return create_react_agent(model=get_agent_llm(), tools=CYCLE_TOOLS, prompt=prompt)

    async def invoke(self, task: AgentTask) -> AgentResult:
        set_user_context(task.user_id, task.workspace_id, task.project_id)
        agent = self._build_agent(self._build_prompt(task))
        messages = task.messages + [HumanMessage(content=task.request)]
        try:
            result = await agent.ainvoke({"messages": messages})
            return AgentResult(content=str(result["messages"][-1].content), agent_name=self.card.name)
        except Exception as exc:
            logger.error("[CycleSubAgent] Error: %s", exc)
            return AgentResult(content=f"Cycle agent error: {exc}", agent_name=self.card.name, has_error=True)

    async def stream(self, task: AgentTask) -> AsyncIterator[dict]:
        set_user_context(task.user_id, task.workspace_id, task.project_id)
        agent = self._build_agent(self._build_prompt(task))
        messages = task.messages + [HumanMessage(content=task.request)]

        async for event in agent.astream_events({"messages": messages}, version="v2"):
            kind = event.get("event", "")
            if kind == "on_tool_start":
                yield {"type": "tool_start", "agent": self.card.name,
                       "tool": event.get("name", ""), "input": event.get("data", {}).get("input", {})}
            elif kind == "on_tool_end":
                out = event.get("data", {}).get("output", "")
                if hasattr(out, "content"):
                    out = out.content
                yield {"type": "tool_end", "agent": self.card.name, "tool": event.get("name", ""), "output": out}
            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    token = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                    if token:
                        yield {"type": "token", "content": token}

        yield {"type": "done", "agent": self.card.name}
