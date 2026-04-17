"""
Orchestrator Agent — A2A Master Coordinator
============================================

The Orchestrator receives requests from the Supervisor pipeline and:
1. Analyzes intent to decide which sub-agent(s) should handle the request
2. Delegates to specialized sub-agents (Task, Project, Cycle, Workspace)
3. Streams events with Agent Handoff notifications for the frontend
4. Handles multi-agent workflows (e.g., create task → assign member)

Architecture:
    Supervisor → [route="action"] → Orchestrator → TaskSubAgent
                                               → ProjectSubAgent
                                               → CycleSubAgent
                                               → WorkspaceSubAgent

The Orchestrator uses LangChain tools that wrap each sub-agent's stream()
so LangGraph's ReAct loop can call them naturally.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from app.agents.card import build_orchestrator_catalog
from app.agents.sub_agents import (
    TaskSubAgent,
    ProjectSubAgent,
    CycleSubAgent,
    WorkspaceSubAgent,
)
from app.agents.card import AgentTask
from app.config import get_agent_llm
from app.models import ChatRequest
from app.tools._client import set_user_context

logger = logging.getLogger(__name__)

# ── Sub-agent singletons ────────────────────────────────────────────────────

_task_agent = TaskSubAgent()
_project_agent = ProjectSubAgent()
_cycle_agent = CycleSubAgent()
_workspace_agent = WorkspaceSubAgent()


# ── Context holder (per-request, set before agent runs) ─────────────────────
# Uses Python contextvars for per-coroutine isolation

from contextvars import ContextVar

_request_ctx: ContextVar[dict] = ContextVar("_request_ctx", default={})


def _make_task(request: str, messages: list[BaseMessage]) -> AgentTask:
    """Build AgentTask from current request context."""
    ctx = _request_ctx.get()
    return AgentTask(
        request=request,
        messages=messages,
        user_id=ctx.get("user_id", ""),
        workspace_id=ctx.get("workspace_id", ""),
        project_id=ctx.get("project_id", ""),
        chat_id=ctx.get("chat_id", ""),
    )


# ── Sub-agent Tool Wrappers ──────────────────────────────────────────────────
# Each sub-agent is exposed as a tool so the Orchestrator's ReAct loop can
# call them. They run the full streaming pipeline and return a text summary.

@tool
async def delegate_to_task_agent(request: str) -> str:
    """Delegate to the Task Agent for task CRUD operations.

    Use when user asks to list, create, update, or delete tasks,
    or wants task statistics / member workload summary.

    Args:
        request: The focused task-related request to handle.
    """
    ctx = _request_ctx.get()
    task = AgentTask(
        request=request,
        user_id=ctx.get("user_id", ""),
        workspace_id=ctx.get("workspace_id", ""),
        project_id=ctx.get("project_id", ""),
    )
    result = await _task_agent.invoke(task)
    return result.content


@tool
async def delegate_to_project_agent(request: str) -> str:
    """Delegate to the Project Agent for project and member management.

    Use when user asks about project overview, progress, team members,
    or wants to update project settings / manage membership.

    Args:
        request: The focused project-related request to handle.
    """
    ctx = _request_ctx.get()
    task = AgentTask(
        request=request,
        user_id=ctx.get("user_id", ""),
        workspace_id=ctx.get("workspace_id", ""),
        project_id=ctx.get("project_id", ""),
    )
    result = await _project_agent.invoke(task)
    return result.content


@tool
async def delegate_to_cycle_agent(request: str) -> str:
    """Delegate to the Cycle Agent for research sprint/cycle management.

    Use when user asks to list cycles, create a new research sprint,
    check cycle progress, or update/delete cycles.

    Args:
        request: The focused cycle-related request to handle.
    """
    ctx = _request_ctx.get()
    task = AgentTask(
        request=request,
        user_id=ctx.get("user_id", ""),
        workspace_id=ctx.get("workspace_id", ""),
        project_id=ctx.get("project_id", ""),
    )
    result = await _cycle_agent.invoke(task)
    return result.content


@tool
async def delegate_to_workspace_agent(request: str) -> str:
    """Delegate to the Workspace Agent for workspace-level queries and analysis.

    Use when user asks about workspace overview, activity feed, global search,
    sticky notes, tags, or analytical reports (workload, overdue, velocity).

    Args:
        request: The focused workspace-related request to handle.
    """
    ctx = _request_ctx.get()
    task = AgentTask(
        request=request,
        user_id=ctx.get("user_id", ""),
        workspace_id=ctx.get("workspace_id", ""),
        project_id=ctx.get("project_id", ""),
    )
    result = await _workspace_agent.invoke(task)
    return result.content


# ── All Orchestrator tools ────────────────────────────────────────────────────

ORCHESTRATOR_TOOLS = [
    delegate_to_task_agent,
    delegate_to_project_agent,
    delegate_to_cycle_agent,
    delegate_to_workspace_agent,
]

# ── System Prompt ─────────────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the **Orchestrator** of Flux-AI — a multi-agent research workspace assistant.

Your job is to understand the user's request and delegate it to the right \
specialized sub-agent. You do NOT handle requests directly; you coordinate.

## Available Sub-Agents
{agent_catalog}

## Context
- Current time: {current_time}
- Workspace ID: {workspace_id}
- Active Project ID: {project_id}
- User ID: {user_id}

{project_note}

## Delegation Rules
1. **Tasks** → `delegate_to_task_agent` (list, create, update, delete tasks; workload)
2. **Projects & Members** → `delegate_to_project_agent` (overview, progress, membership)
3. **Cycles/Sprints** → `delegate_to_cycle_agent` (sprint management)
4. **Workspace / Analysis** → `delegate_to_workspace_agent` (overview, activity, search, stickies, reports)

## Multi-Agent Workflows
For complex requests requiring multiple agents, call them sequentially:
- "tạo task và assign cho member A" → delegate_to_task_agent (create) then clarify with result
- "tóm tắt tiến độ cả project" → delegate_to_project_agent (overview) + delegate_to_task_agent (tasks)

## Rules
1. Always delegate — never try to answer directly from your training data.
2. Pass a clear, focused request to each sub-agent.
3. After all delegations, synthesize the results into a coherent response.
4. Respond in the **same language** as the user.
5. Use markdown formatting for structured output.
"""


def _build_orchestrator_prompt(
    workspace_id: str,
    project_id: str,
    user_id: str,
) -> str:
    project_note = (
        f"⚡ **Active Project**: `{project_id}` — pass this as context when delegating to Task/Project agents.\n"
        if project_id
        else "ℹ️ No active project — ask user to select one if project-specific info is needed.\n"
    )
    return ORCHESTRATOR_SYSTEM_PROMPT.format(
        agent_catalog=build_orchestrator_catalog(),
        current_time=datetime.now(timezone.utc).isoformat(),
        workspace_id=workspace_id or "N/A",
        project_id=project_id or "none",
        user_id=user_id or "unknown",
        project_note=project_note,
    )


# ── Streaming Orchestrator ─────────────────────────────────────────────────────

# Map tool name → sub-agent name for labeling
_TOOL_TO_AGENT: dict[str, str] = {
    "delegate_to_task_agent": "task_agent",
    "delegate_to_project_agent": "project_agent",
    "delegate_to_cycle_agent": "cycle_agent",
    "delegate_to_workspace_agent": "workspace_agent",
}

CONFIRM_REQUIRED_TOOLS = {
    "delegate_to_task_agent",
    "delegate_to_project_agent",
    "delegate_to_cycle_agent",
}


def _extract_token(content) -> str:
    """Extract plain text from a LangChain chunk content.

    Handles str, list-of-blocks (Groq/Anthropic structured format), and
    silently ignores non-text block types (tool_use, thinking, signature).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""



async def stream_orchestrator(
    messages: list[BaseMessage],
    request: ChatRequest,
) -> AsyncIterator[str]:
    """Stream SSE events from the Orchestrator and its sub-agents.

    SSE events emitted:
      - data: [ACTION]{"type": "agent_handoff", "to": "task_agent", ...}
      - data: [ACTION]{"type": "tool_start", "agent": "task_agent", ...}
      - data: [ACTION]{"type": "tool_end", ...}
      - data: <token>
    """
    user_id = request.user_id or ""
    workspace_id = request.workspace_id or ""
    project_id = request.project_id or ""
    chat_id = request.chat_id or ""

    # Set context vars for sub-agent tool wrappers
    set_user_context(user_id, workspace_id, project_id)
    _request_ctx.set({
        "user_id": user_id,
        "workspace_id": workspace_id,
        "project_id": project_id,
        "chat_id": chat_id,
    })

    system_prompt = _build_orchestrator_prompt(workspace_id, project_id, user_id)
    agent_llm = get_agent_llm()
    orchestrator = create_react_agent(
        model=agent_llm,
        tools=ORCHESTRATOR_TOOLS,
        prompt=system_prompt,
    )

    in_tool_call = 0
    async for event in orchestrator.astream_events({"messages": messages}, version="v2"):
        kind = event.get("event", "")

        # ── Orchestrator calls a sub-agent tool ──────────────────────────────
        if kind == "on_tool_start":
            in_tool_call += 1
            tool_name = event.get("name", "")
            tool_input = event.get("data", {}).get("input", {})
            agent_target = _TOOL_TO_AGENT.get(tool_name, tool_name)

            # Only emit AGENT_HANDOFF for our known sub-agent tools
            if tool_name in _TOOL_TO_AGENT:
                handoff_event = {
                    "type": "agent_handoff",
                    "from": "orchestrator",
                    "to": agent_target,
                    "request": tool_input.get("request", "")[:200],
                }
                yield f"data: [ACTION]{json.dumps(handoff_event)}\n\n"
                logger.info("Orchestrator → %s", agent_target)

        # ── Sub-agent returned ────────────────────────────────────────────────
        elif kind == "on_tool_end":
            tool_name = event.get("name", "")
            agent_target = _TOOL_TO_AGENT.get(tool_name, tool_name)

            if tool_name in _TOOL_TO_AGENT:
                yield f"data: [ACTION]{json.dumps({'type': 'agent_done', 'agent': agent_target, 'success': True})}\n\n"
                logger.info("Sub-agent done: %s", agent_target)

            # Decrement AFTER processing the end event
            in_tool_call = max(0, in_tool_call - 1)

        # ── LLM streaming token ───────────────────────────────────────────────
        # GUARD: only process when NOT inside a tool call.
        # When in_tool_call > 0, these events come from nested sub-agent LLMs
        # (bubbled up by astream_events v2). Streaming them would duplicate the
        # content that the Orchestrator synthesizes in its final response.
        elif kind == "on_chat_model_stream" and in_tool_call == 0:
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                # Skip chunks that are tool-calling decisions (not final text)
                additional = getattr(chunk, "additional_kwargs", {})
                if additional.get("tool_calls"):
                    continue
                token = _extract_token(chunk.content)
                if token:
                    yield f"data: {token.replace(chr(10), chr(92) + 'n')}\n\n"

