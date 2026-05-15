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

from app.agents.card import AgentCard, AgentTask, AgentResult, register_card  # noqa: F401
from app.agents.sub_agents.base_sub_agent import BaseSubAgent
from app.tools.task.task_tools import (
    list_tasks,
    get_my_tasks,
    summarize_member_tasks,
    create_task,
    update_task,
    delete_task,
)

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

# ── System Prompt ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
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


# ── Sub-Agent ──────────────────────────────────────────────────────────────

class TaskSubAgent(BaseSubAgent):
    """Task-focused sub-agent invoked by the Orchestrator."""

    card = TASK_AGENT_CARD
    tools = [
        list_tasks, get_my_tasks, summarize_member_tasks,
        create_task, update_task, delete_task,
    ]

    def _build_prompt(self, task: AgentTask) -> str:
        project_id = task.project_id
        project_note = (
            f"⚡ **Active Project**: `{project_id}` — use this ID for task operations.\n"
            if project_id
            else "ℹ️ No project selected — ask the Orchestrator to provide one if needed.\n"
        )
        return _SYSTEM_PROMPT.format(
            workspace_id=task.workspace_id or "N/A",
            project_id=project_id or "none",
            user_id=task.user_id or "unknown",
            project_note=project_note,
        )
