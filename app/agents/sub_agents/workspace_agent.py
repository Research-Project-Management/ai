"""
Workspace Sub-Agent — Handles workspace-level operations & analysis.

Invoked by Orchestrator when the user wants:
- Workspace overview (recent items, all projects summary)
- Activity feed (who did what recently)
- Global search across workspace
- Stickies, tags, page listing
- Analysis: workload distribution, overdue report, velocity, team summary
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from app.agents.card import AgentCard, AgentTask, AgentResult, register_card
from app.config import get_agent_llm
from app.tools._client import set_user_context
from app.tools.workspace_tools import (
    search_workspace,
    list_members,
    get_workspace_overview,
    get_workspace_activity,
)
from app.tools.sticky_tools import list_stickies, create_sticky, update_sticky, delete_sticky
from app.tools.tag_tools import list_tags, create_tag, delete_tag
from app.tools.page_tools import list_pages, list_workspace_pages, get_page_content
from app.tools.user_tools import get_current_user, search_users
from app.tools.analysis_tools import (
    get_workload_distribution,
    get_overdue_report,
    get_project_velocity,
    generate_team_summary,
)

logger = logging.getLogger(__name__)

WORKSPACE_AGENT_CARD = register_card(AgentCard(
    name="workspace_agent",
    description=(
        "Handles workspace-level queries: overview (all projects, recent activity), "
        "global search, sticky notes, tags, workspace members, and analytical reports "
        "(workload distribution, overdue report, team velocity)."
    ),
    skills=[
        "get_workspace_overview", "get_workspace_activity", "search_workspace",
        "list_members", "list_stickies", "create_sticky",
        "get_workload_distribution", "get_overdue_report", "generate_team_summary",
    ],
    has_destructive_ops=False,
))

WORKSPACE_TOOLS = [
    search_workspace, list_members, get_workspace_overview, get_workspace_activity,
    list_stickies, create_sticky, update_sticky, delete_sticky,
    list_tags, create_tag, delete_tag,
    list_workspace_pages, list_pages, get_page_content,
    get_current_user, search_users,
    get_workload_distribution, get_overdue_report, get_project_velocity, generate_team_summary,
]

WORKSPACE_SYSTEM_PROMPT = """\
You are the **Workspace Agent** — a specialized AI for workspace-level insights \
and management in Flux-AI.

## Your Scope
- Workspace overviews, activity feeds, global search
- Sticky notes management
- Tags management
- Analytical reports (workload, overdue, velocity, team summary)
- User lookup

You do NOT handle task CRUD or project member management (other agents handle those).

## Context
- Workspace ID: {workspace_id}
- User ID: {user_id}

## Rules
1. For "workspace có gì?" → use get_workspace_overview.
2. For "có gì mới?" → use get_workspace_activity.
3. For analysis requests → use the analysis tools.
4. Always respond in the same language as the user.
5. Use markdown for reports and tables.
"""


class WorkspaceSubAgent:
    card = WORKSPACE_AGENT_CARD

    def _build_prompt(self, task: AgentTask) -> str:
        return WORKSPACE_SYSTEM_PROMPT.format(
            workspace_id=task.workspace_id or "N/A",
            user_id=task.user_id or "unknown",
        )

    def _build_agent(self, prompt: str):
        return create_react_agent(model=get_agent_llm(), tools=WORKSPACE_TOOLS, prompt=prompt)

    async def invoke(self, task: AgentTask) -> AgentResult:
        set_user_context(task.user_id, task.workspace_id, task.project_id)
        agent = self._build_agent(self._build_prompt(task))
        messages = task.messages + [HumanMessage(content=task.request)]
        try:
            result = await agent.ainvoke({"messages": messages})
            return AgentResult(content=str(result["messages"][-1].content), agent_name=self.card.name)
        except Exception as exc:
            logger.error("[WorkspaceSubAgent] Error: %s", exc)
            return AgentResult(content=f"Workspace agent error: {exc}", agent_name=self.card.name, has_error=True)

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
