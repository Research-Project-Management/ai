"""
Project Sub-Agent — Handles project & member management.

Invoked by Orchestrator when the user wants to:
- List/view projects
- Get project overview/progress
- Manage project members (add, remove, update role)
- Update project settings
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from app.agents.card import AgentCard, AgentTask, AgentResult, register_card
from app.config import get_agent_llm
from app.tools._client import set_user_context
from app.tools.project_tools import (
    list_projects,
    get_project_overview,
    get_project_details,
    update_project,
    add_project_member,
    remove_project_member,
    update_member_role,
)
from app.tools.workspace_tools import list_members

logger = logging.getLogger(__name__)

# ── Agent Card ─────────────────────────────────────────────────────────────

PROJECT_AGENT_CARD = register_card(AgentCard(
    name="project_agent",
    description=(
        "Manages projects and team members: listing projects, viewing project overview, "
        "progress reports, adding/removing members, updating project settings."
    ),
    skills=[
        "list_projects", "get_project_overview", "get_project_details",
        "update_project", "add_project_member", "remove_project_member",
        "update_member_role", "list_members",
    ],
    has_destructive_ops=True,
))

PROJECT_TOOLS = [
    list_projects,
    get_project_overview,
    get_project_details,
    update_project,
    add_project_member,
    remove_project_member,
    update_member_role,
    list_members,
]

PROJECT_SYSTEM_PROMPT = """\
You are the **Project Agent** — a specialized AI for managing projects and team \
members in the user's research workspace.

## Your Scope
You handle: project listing, project overview/progress, member management, \
and project settings. You do NOT handle task CRUD (the Task Agent handles that).

## Context
- Workspace ID: {workspace_id}
- Active Project ID: {project_id}
- User ID: {user_id}

## Rules
1. For member operations (add/remove/update role), always confirm the member's \
   user_id by calling list_members first.
2. For progress reports, use get_project_overview for stats.
3. Always respond in the same language as the user.
4. Use markdown tables for member lists and progress summaries.
"""


class ProjectSubAgent:
    """Project & member management sub-agent."""

    card = PROJECT_AGENT_CARD

    def _build_prompt(self, task: AgentTask) -> str:
        return PROJECT_SYSTEM_PROMPT.format(
            workspace_id=task.workspace_id or "N/A",
            project_id=task.project_id or "none",
            user_id=task.user_id or "unknown",
        )

    def _build_agent(self, system_prompt: str):
        return create_react_agent(
            model=get_agent_llm(),
            tools=PROJECT_TOOLS,
            prompt=system_prompt,
        )

    async def invoke(self, task: AgentTask) -> AgentResult:
        set_user_context(task.user_id, task.workspace_id, task.project_id)
        system_prompt = self._build_prompt(task)
        agent = self._build_agent(system_prompt)
        messages = task.messages + [HumanMessage(content=task.request)]

        try:
            result = await agent.ainvoke({"messages": messages})
            content = result["messages"][-1].content
            return AgentResult(content=str(content), agent_name=self.card.name)
        except Exception as exc:
            logger.error("[ProjectSubAgent] Error: %s", exc, exc_info=True)
            return AgentResult(
                content=f"Project agent error: {exc}",
                agent_name=self.card.name,
                has_error=True,
            )

    async def stream(self, task: AgentTask) -> AsyncIterator[dict]:
        set_user_context(task.user_id, task.workspace_id, task.project_id)
        system_prompt = self._build_prompt(task)
        agent = self._build_agent(system_prompt)
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
                yield {"type": "tool_end", "agent": self.card.name,
                       "tool": event.get("name", ""), "output": out}
            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    token = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                    if token:
                        yield {"type": "token", "content": token}

        yield {"type": "done", "agent": self.card.name}
