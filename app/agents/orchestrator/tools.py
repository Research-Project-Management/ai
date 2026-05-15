"""
Orchestrator delegate tools.

These LangChain tools wrap the sub-agents' stream() methods. They forward
real-time token events to the shared queue and return the final text
so the Orchestrator LLM can synthesize a final answer.
"""

import asyncio
import logging

from langchain_core.tools import tool

from app.rag.retriever import retrieve_documents
from app.agents.card import AgentTask
from app.agents.sub_agents import (
    TaskSubAgent,
    ProjectSubAgent,
    CycleSubAgent,
    WorkspaceSubAgent,
)
from app.agents.orchestrator.context import _request_ctx, _sub_agent_event_queue
from app.tools._client import get_trace_id

logger = logging.getLogger(__name__)

# ── Sub-agent singletons ────────────────────────────────────────────────────

_task_agent = TaskSubAgent()
_project_agent = ProjectSubAgent()
_cycle_agent = CycleSubAgent()
_workspace_agent = WorkspaceSubAgent()

# Default timeout per sub-agent stream
_SUB_AGENT_TIMEOUT = 60.0  # seconds


async def _delegate(agent, request: str, label: str, timeout: float = _SUB_AGENT_TIMEOUT) -> str:
    """Generic streaming delegate: stream sub-agent, forward events to queue."""
    ctx = _request_ctx.get()
    task = AgentTask(
        request=request,
        user_id=ctx.get("user_id", ""),
        workspace_id=ctx.get("workspace_id", ""),
        project_id=ctx.get("project_id", ""),
    )
    queue = _sub_agent_event_queue.get()
    content_parts: list[str] = []
    stream_error: str | None = None

    try:
        async with asyncio.timeout(timeout):
            async for event in agent.stream(task):
                if event.get("type") == "token":
                    content_parts.append(event.get("content", ""))
                if queue is not None:
                    await queue.put(("sub_agent", event))
    except TimeoutError:
        logger.warning(
            "[Orchestrator] %s timed out after %.0fs — returning partial result",
            label, timeout,
        )
        partial = "".join(content_parts)
        if partial:
            return partial + f"\n\n> ⚠️ *{label} response was cut short due to timeout.*"
        return f"[ERROR] {label} timed out after {timeout:.0f}s. Please try a simpler or more specific request."
    except Exception as exc:
        logger.error("[Orchestrator] %s stream error: %s", label, exc, exc_info=True)
        stream_error = str(exc)

    if stream_error:
        return f"[ERROR] {label} failed: {stream_error}. Please inform the user and suggest retrying."

    content = "".join(content_parts)
    if not content:
        return f"[ERROR] {label} returned empty response. Please try rephrasing."
    return content


@tool
async def delegate_to_task_agent(request: str) -> str:
    """Delegate to the Task Agent for task CRUD operations.

    Use when user asks to list, create, update, or delete tasks,
    or wants task statistics / member workload summary.
    """
    return await _delegate(_task_agent, request, "Task agent")


@tool
async def delegate_to_project_agent(request: str) -> str:
    """Delegate to the Project Agent for project and member management.

    Use when user asks about project overview, progress, team members,
    or wants to update project settings / manage membership.
    """
    return await _delegate(_project_agent, request, "Project agent")


@tool
async def delegate_to_cycle_agent(request: str) -> str:
    """Delegate to the Cycle Agent for research sprint/cycle management.

    Use when user asks to list cycles, create a new research sprint,
    check cycle progress, or update/delete cycles.
    """
    return await _delegate(_cycle_agent, request, "Cycle agent")


@tool
async def delegate_to_workspace_agent(request: str) -> str:
    """Delegate to the Workspace Agent for workspace-level queries and analysis.

    Use when user asks about workspace overview, activity feed, global search,
    sticky notes, tags, or analytical reports (workload, overdue, velocity).
    """
    return await _delegate(_workspace_agent, request, "Workspace agent")


@tool
async def delegate_parallel(
    task_request: str | None = None,
    project_request: str | None = None,
    cycle_request: str | None = None,
    workspace_request: str | None = None,
) -> str:
    """Delegate to MULTIPLE sub-agents IN PARALLEL for independent requests.

    Use when the user asks for information from 2+ independent domains at once.
    Results are combined and returned together.
    """
    coros = []
    labels = []

    if task_request:
        coros.append(_delegate(_task_agent, task_request, "Task agent"))
        labels.append("## Tasks")
    if project_request:
        coros.append(_delegate(_project_agent, project_request, "Project agent"))
        labels.append("## Project")
    if cycle_request:
        coros.append(_delegate(_cycle_agent, cycle_request, "Cycle agent"))
        labels.append("## Cycles")
    if workspace_request:
        coros.append(_delegate(_workspace_agent, workspace_request, "Workspace agent"))
        labels.append("## Workspace")

    if not coros:
        return "[ERROR] delegate_parallel called with no requests."

    logger.info("[%s] Parallel delegation: %d agents", get_trace_id(), len(coros))

    results = await asyncio.gather(*coros, return_exceptions=True)

    parts = []
    for label, result in zip(labels, results):
        if isinstance(result, Exception):
            logger.warning("[%s] Parallel sub-agent failed: %s", get_trace_id(), result)
            parts.append(f"{label}\n[ERROR] {result}")
        else:
            parts.append(f"{label}\n{result}")

    return "\n\n---\n\n".join(parts)


@tool
async def query_uploaded_documents(query: str) -> str:
    """Query the user's uploaded documents/CVs.
    
    Use this when you need to read or extract information from the documents 
    the user has uploaded (like parsing a CV for skills to create tasks).
    """
    ctx = _request_ctx.get()
    document_ids = ctx.get("document_ids", [])
    if not document_ids:
        return "[ERROR] No documents are currently selected or uploaded in this chat session."
    
    chunks = await retrieve_documents(
        query=query,
        k=8,
        document_ids=document_ids,
        user_id=ctx.get("user_id"),
        chat_id=ctx.get("chat_id"),
    )
    
    if not chunks:
        return "No relevant information found in the uploaded documents."
    
    parts = []
    for c in chunks:
        parts.append(f"Source: {c['source']}\n{c['content']}")
        
    return "\n\n---\n\n".join(parts)



ORCHESTRATOR_TOOLS = [
    delegate_to_task_agent,
    delegate_to_project_agent,
    delegate_to_cycle_agent,
    delegate_to_workspace_agent,
    delegate_parallel,
    query_uploaded_documents,
]

CONFIRM_REQUIRED_TOOLS = {
    "delegate_to_task_agent",
    "delegate_to_project_agent",
    "delegate_to_cycle_agent",
}
