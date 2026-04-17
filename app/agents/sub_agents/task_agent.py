"""
Task Sub-Agent — Handles all task-related operations.

This agent is invoked by the Orchestrator when the user wants to:
- List, create, update, or delete tasks
- Get task statistics / overdue report
- Summarize a member's work

Tools exposed: list_tasks, get_my_tasks, summarize_member_tasks,
               create_task, update_task, delete_task
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from app.agents.card import AgentCard, AgentTask, AgentResult, register_card
from app.config import get_agent_llm
from app.tools._client import set_user_context
from app.tools.task_tools import (
    list_tasks,
    get_my_tasks,
    summarize_member_tasks,
    create_task,
    update_task,
    delete_task,
)

logger = logging.getLogger(__name__)

# ── Agent Card ─────────────────────────────────────────────────────────────

TASK_AGENT_CARD = register_card(AgentCard(
    name="task_agent",
    description=(
        "Handles task management: listing, creating, updating, deleting tasks. "
        "Use when user asks about tasks, workload, overdue items, or wants to "
        "assign/track work within a project."
    ),
    skills=[
        "list_tasks", "get_my_tasks", "summarize_member_tasks",
        "create_task", "update_task", "delete_task",
    ],
    has_destructive_ops=True,
))

# ── Tools for this agent only ──────────────────────────────────────────────

TASK_TOOLS = [
    list_tasks,
    get_my_tasks,
    summarize_member_tasks,
    create_task,
    update_task,
    delete_task,
]

# ── System Prompt ──────────────────────────────────────────────────────────

TASK_SYSTEM_PROMPT = """\
You are the **Task Agent** — a specialized AI that manages tasks in the user's \
research workspace.

## Your Scope
You ONLY handle task-related operations. Do NOT attempt to manage projects, \
members, cycles, or pages (the Orchestrator handles routing).

## Context
- Workspace ID: {workspace_id}
- Active Project ID: {project_id}
- User ID: {user_id}

{project_note}

## Output Formatting Rules (MANDATORY)
Always use **rich Markdown** — never plain text lists.

### For task lists, use this format:
```
## 📋 [Project/Section Name]

### 📌 Backlog
| # | Task | Priority | Due |
|---|------|----------|-----|
| 1 | **Task title** | 🔴 Urgent | 2026-05-01 |

### 🔄 In Progress
| # | Task | Priority | Assignee |
|---|------|----------|----------|
| 1 | **Task title** | 🟡 Medium | @name |

### ✅ Done
...

---
**Summary:** X tasks total — Y in progress, Z done, W overdue 🔴
```

### Priority emoji mapping:
- 🔴 **urgent** — phải làm ngay
- 🟠 **high** — ưu tiên cao
- 🟡 **medium** — bình thường
- 🔵 **low** — thấp
- ⚪ **none** — chưa xác định

### Column/Status mapping:
- 📌 Backlog = chưa bắt đầu
- 📋 To Do = cần làm
- 🔄 In Progress = đang làm
- 👀 In Review = đang review
- ✅ Done = hoàn thành

### For overdue tasks, add ⏰ before the title.
### For "my tasks" (get_my_tasks), group by project then by status.

## Tool Usage
- `list_tasks(project_id)` — list all tasks in a project
- `get_my_tasks(workspace_id)` — get tasks assigned to current user
- `create_task(project_id, title, column_id, ...)` — create a new task
- `update_task(task_id, ...)` — update task status, priority, assignee
- `delete_task(task_id)` — delete a task (always confirm first)
- `summarize_member_tasks(project_id, member_id)` — workload summary

## General Rules
1. If project_id is available above, use it directly — do NOT call list_projects.
2. Always respond in the **same language** as the user.
3. After listing tasks, add a 1-line insight (e.g., "3 tasks are overdue 🔴").
"""



# ── Sub-Agent Class ────────────────────────────────────────────────────────

class TaskSubAgent:
    """Task-focused sub-agent invoked by the Orchestrator."""

    card = TASK_AGENT_CARD

    def _build_prompt(self, task: AgentTask) -> str:
        project_id = task.project_id
        project_note = (
            f"⚡ **Active Project**: `{project_id}` — use this ID for task operations.\n"
            if project_id
            else "ℹ️ No project selected — ask the Orchestrator to provide one if needed.\n"
        )
        return TASK_SYSTEM_PROMPT.format(
            workspace_id=task.workspace_id or "N/A",
            project_id=project_id or "none",
            user_id=task.user_id or "unknown",
            project_note=project_note,
        )

    def _build_agent(self, system_prompt: str):
        """Create a fresh ReAct agent with task tools only."""
        return create_react_agent(
            model=get_agent_llm(),
            tools=TASK_TOOLS,
            prompt=system_prompt,
        )

    async def invoke(self, task: AgentTask) -> AgentResult:
        """Run the task agent (non-streaming)."""
        set_user_context(task.user_id, task.workspace_id, task.project_id)
        system_prompt = self._build_prompt(task)
        agent = self._build_agent(system_prompt)

        messages = task.messages + [HumanMessage(content=task.request)]
        tool_calls: list[dict] = []

        try:
            result = await agent.ainvoke({"messages": messages})
            final_message = result["messages"][-1]
            content = (
                final_message.content
                if isinstance(final_message.content, str)
                else str(final_message.content)
            )
            return AgentResult(
                content=content,
                agent_name=self.card.name,
                tool_calls=tool_calls,
            )
        except Exception as exc:
            logger.error("[TaskSubAgent] Error: %s", exc, exc_info=True)
            return AgentResult(
                content=f"Task agent encountered an error: {exc}",
                agent_name=self.card.name,
                has_error=True,
            )

    async def stream(self, task: AgentTask) -> AsyncIterator[dict]:
        """Stream events from the task agent.

        Yields dicts with 'type' and relevant fields:
          - {"type": "tool_start", "tool": ..., "input": ...}
          - {"type": "tool_end", "tool": ..., "output": ...}
          - {"type": "token", "content": ...}
          - {"type": "done", "agent": ...}
        """
        set_user_context(task.user_id, task.workspace_id, task.project_id)
        system_prompt = self._build_prompt(task)
        agent = self._build_agent(system_prompt)

        messages = task.messages + [HumanMessage(content=task.request)]

        async for event in agent.astream_events({"messages": messages}, version="v2"):
            kind = event.get("event", "")

            if kind == "on_tool_start":
                yield {
                    "type": "tool_start",
                    "agent": self.card.name,
                    "tool": event.get("name", ""),
                    "input": event.get("data", {}).get("input", {}),
                }

            elif kind == "on_tool_end":
                output = event.get("data", {}).get("output", "")
                if hasattr(output, "content"):
                    output = output.content
                yield {
                    "type": "tool_end",
                    "agent": self.card.name,
                    "tool": event.get("name", ""),
                    "output": output,
                }

            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    token = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                    if token:
                        yield {"type": "token", "content": token}

        yield {"type": "done", "agent": self.card.name}
