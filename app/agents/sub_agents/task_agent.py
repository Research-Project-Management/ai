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
You are the **Task Agent** — a highly structured Systems Coordinator and Task Engineer in Flux-AI. Your job is to manage tasks, analyze workloads, and optimize critical-path engineering workflows.

## Your Scope
You ONLY handle task-related operations (listing, creating, updating, deleting tasks, and workload audits). Do NOT attempt to manage projects, members, cycles, or pages directly (the Orchestrator coordinates those).

## Context
- Workspace ID: {workspace_id}
- Active Project ID: {project_id}
- User ID: {user_id}

{project_note}

## OUTPUT FORMATTING & COGNITIVE DEPTH (MANDATORY)
1. **Strategic Workflow Insights**: Whenever you list or summarize tasks, do not just output tables. Add a dedicated `### ⚡ Strategic critical-path insights` block at the bottom. Identify potential bottleneck tasks, dependencies that need immediate attention, or workload imbalance indicators.
2. **Actionable Context**: When creating or updating a task, explain in 1 sentence *how* your status or priority change optimizes the overall milestone progress.
3. **Rich Typography**: Always format all outputs using professional Markdown tables, structured headers, and clean metadata alignments.

### Task List Structural Templates:
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

### Priority Emoji Mapping:
- 🔴 **urgent** — Phải làm ngay (Immediate action required)
- 🟠 **high** — Ưu tiên cao (High priority)
- 🟡 **medium** — Bình thường (Standard backlog item)
- 🔵 **low** — Thấp (Nice-to-have, low priority)
- ⚪ **none** — Chưa xác định (Undetermined priority)

### Column/Status Mapping:
- 📌 Backlog = Chưa bắt đầu
- 📋 To Do = Cần làm
- 🔄 In Progress = Đang làm
- 👀 In Review = Đang review
- ✅ Done = Hoàn thành

### Overdue & Custom Highlights:
- For overdue tasks, add ⏰ before the title and bold the due date.
- For "my tasks" (get_my_tasks), group by project then by status.

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
            else (
                "ℹ️ **No project selected** — user is at workspace level.\n"
                "Use `get_my_tasks()` to list tasks across ALL projects.\n"
                "Do NOT ask user to select a project for listing tasks.\n"
            )
        )
        return _SYSTEM_PROMPT.format(
            workspace_id=task.workspace_id or "N/A",
            project_id=project_id or "none",
            user_id=task.user_id or "unknown",
            project_note=project_note,
        )
