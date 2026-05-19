"""
Streaming Orchestrator Engine.

Runs the ReAct graph, routes the sub-agent token streams, and synthesizes
final output while emitting frontend action events.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import AsyncIterator

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from app.agents.card import build_orchestrator_catalog
from app.agents.conversation_context import build_conversation_memory
from app.agents.orchestrator.context import _request_ctx, _sub_agent_event_queue
from app.agents.orchestrator.tools import ORCHESTRATOR_TOOLS
from app.config import get_agent_llm
from app.models import ChatRequest
from app.tools._client import get_trace_id
from app.agents.utils import _extract_token

logger = logging.getLogger(__name__)

# ── Pre-compiled ReAct graph ──────────────────────────────────────────────────

def _dynamic_prompt(state: dict) -> list:
    ctx = _request_ctx.get()
    system_prompt = ctx.get("system_prompt", "")
    messages = state.get("messages", [])
    if system_prompt:
        return [SystemMessage(content=system_prompt)] + list(messages)
    return list(messages)


_agent_llm = get_agent_llm()
_ORCHESTRATOR_GRAPH = create_react_agent(
    model=_agent_llm,
    tools=ORCHESTRATOR_TOOLS,
    prompt=_dynamic_prompt,
)

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
{conversation_context}

## Delegation Rules
1. **Tasks** → `delegate_to_task_agent` (list, create, update, delete tasks; workload)
2. **Projects & Members** → `delegate_to_project_agent` (overview, progress, membership)
3. **Cycles/Sprints** → `delegate_to_cycle_agent` (sprint management)
4. **Workspace / Analysis** → `delegate_to_workspace_agent` (overview, activity, search, stickies, reports)
5. **Multiple independent domains** → `delegate_parallel` (runs sub-agents simultaneously, 2-4x faster)
6. **Read Uploaded Documents** → `query_uploaded_documents` (Use this FIRST to extract information if the user asks to generate tasks from an uploaded CV or document).

## When to Use delegate_parallel
Use `delegate_parallel` when the user asks for information from **2+ independent domains** at once:
- "tasks + workspace overview" → `delegate_parallel(task_request=..., workspace_request=...)`
- "project members + cycles" → `delegate_parallel(project_request=..., cycle_request=...)`
- "tóm tắt workspace và tasks" → `delegate_parallel(workspace_request=..., task_request=...)`

Do NOT use `delegate_parallel` when results depend on each other (e.g., create task then assign).

## Sequential Multi-Agent Workflows
For dependent workflows, call agents sequentially:
- "tạo task và assign cho member A" → delegate_to_task_agent (create) then clarify with result
- "tìm member theo tên rồi add vào project" → delegate_to_workspace_agent (search) → delegate_to_project_agent (add)

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
    conversation_context: str = "",
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
        conversation_context=(
            "## Conversation Memory\n"
            + conversation_context.strip()
            + "\n\nUse this to resolve follow-up requests before delegating.\n"
            if conversation_context.strip()
            else ""
        ),
    )


# ── Streaming Orchestrator ─────────────────────────────────────────────────────

_TOOL_TO_AGENT: dict[str, str] = {
    "delegate_to_task_agent": "task_agent",
    "delegate_to_project_agent": "project_agent",
    "delegate_to_cycle_agent": "cycle_agent",
    "delegate_to_workspace_agent": "workspace_agent",
}

_PARALLEL_KEY_TO_AGENT: dict[str, str] = {
    "task_request": "task_agent",
    "project_request": "project_agent",
    "cycle_request": "cycle_agent",
    "workspace_request": "workspace_agent",
}

_SENTINEL = object()


async def stream_orchestrator(
    messages: list[BaseMessage],
    request: ChatRequest,
    conversation_memory: str = "",
) -> AsyncIterator[str]:
    """Stream SSE events from the Orchestrator and its sub-agents."""
    user_id = request.user_id or ""
    workspace_id = request.workspace_id or ""
    project_id = request.project_id or ""
    chat_id = request.chat_id or ""

    conversation_context = conversation_memory or build_conversation_memory(messages)
    system_prompt = _build_orchestrator_prompt(
        workspace_id,
        project_id,
        user_id,
        conversation_context,
    )

    _request_ctx.set({
        "user_id": user_id,
        "workspace_id": workspace_id,
        "project_id": project_id,
        "chat_id": chat_id,
        "document_ids": request.document_ids or [],
        "system_prompt": system_prompt,
    })

    event_queue: asyncio.Queue = asyncio.Queue()
    _sub_agent_event_queue.set(event_queue)

    async def _run_graph() -> None:
        try:
            async for event in _ORCHESTRATOR_GRAPH.astream_events(
                {"messages": messages}, version="v2"
            ):
                await event_queue.put(("graph", event))
        except Exception as exc:
            logger.error("[Orchestrator] Graph task error: %s", exc, exc_info=True)
            await event_queue.put(("error", exc))
        finally:
            await event_queue.put(("done", _SENTINEL))

    graph_task = asyncio.create_task(_run_graph())
    in_tool_call = 0
    _sub_agent_streamed = False

    try:
        while True:
            source, payload = await event_queue.get()

            if source == "done":
                break
            if source == "error":
                logger.error("[Orchestrator] Received error from graph task: %s", payload)
                break

            if source == "sub_agent":
                evt = payload
                evt_type = evt.get("type", "")

                if evt_type == "token":
                    token = evt.get("content", "")
                    if token:
                        _sub_agent_streamed = True
                        yield f"data: {token.replace(chr(10), chr(92) + 'n')}\n\n"

                elif evt_type == "tool_start":
                    tool_event = {
                        "type": "tool_call",
                        "agent": evt.get("agent", ""),
                        "tool": evt.get("tool", ""),
                        "status": "calling",
                    }
                    yield f"data: [ACTION]{json.dumps(tool_event)}\n\n"

                elif evt_type == "tool_end":
                    output = evt.get("output", "")
                    has_error = (isinstance(output, dict) and bool(output.get("error"))) or \
                                (isinstance(output, str) and output.startswith("[ERROR]"))
                    tool_event = {
                        "type": "tool_call",
                        "agent": evt.get("agent", ""),
                        "tool": evt.get("tool", ""),
                        "status": "error" if has_error else "done",
                    }
                    yield f"data: [ACTION]{json.dumps(tool_event)}\n\n"
                continue

            event = payload
            kind = event.get("event", "")

            if kind == "on_tool_start":
                in_tool_call += 1
                tool_name = event.get("name", "")
                tool_input = event.get("data", {}).get("input", {})
                agent_target = _TOOL_TO_AGENT.get(tool_name, tool_name)

                if tool_name in _TOOL_TO_AGENT:
                    handoff_event = {
                        "type": "agent_handoff",
                        "from": "orchestrator",
                        "to": agent_target,
                        "request": tool_input.get("request", "")[:200],
                    }
                    yield f"data: [ACTION]{json.dumps(handoff_event)}\n\n"
                    logger.info("[%s] Orchestrator → %s", get_trace_id(), agent_target)

                elif tool_name == "delegate_parallel":
                    active_agents = [
                        _PARALLEL_KEY_TO_AGENT[k]
                        for k in _PARALLEL_KEY_TO_AGENT
                        if tool_input.get(k)
                    ]
                    for ag in active_agents:
                        req_key = ag.replace("_agent", "_request")
                        handoff_event = {
                            "type": "agent_handoff",
                            "from": "orchestrator",
                            "to": ag,
                            "parallel": True,
                            "request": tool_input.get(req_key, "")[:200],
                        }
                        yield f"data: [ACTION]{json.dumps(handoff_event)}\n\n"
                    logger.info("[%s] Orchestrator → parallel %s", get_trace_id(), active_agents)

            elif kind == "on_tool_end":
                tool_name = event.get("name", "")
                agent_target = _TOOL_TO_AGENT.get(tool_name, tool_name)

                if tool_name in _TOOL_TO_AGENT:
                    output = event.get("data", {}).get("output", "")
                    has_error = isinstance(output, str) and output.startswith("[ERROR]")
                    yield f"data: [ACTION]{json.dumps({'type': 'agent_done', 'agent': agent_target, 'success': not has_error})}\n\n"
                    if has_error:
                        logger.warning("[%s] Sub-agent %s error", get_trace_id(), agent_target)
                    else:
                        logger.info("[%s] Sub-agent %s done", get_trace_id(), agent_target)

                in_tool_call = max(0, in_tool_call - 1)

            elif kind == "on_chat_model_stream" and in_tool_call == 0:
                if _sub_agent_streamed:
                    continue
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    additional = getattr(chunk, "additional_kwargs", {})
                    if additional.get("tool_calls"):
                        continue
                    token = _extract_token(chunk.content)
                    if token:
                        yield f"data: {token.replace(chr(10), chr(92) + 'n')}\n\n"

    finally:
        if not graph_task.done():
            graph_task.cancel()
            try:
                await graph_task
            except asyncio.CancelledError:
                pass
