"""Task management tools — CRUD operations on project tasks."""

from __future__ import annotations

import re

from langchain_core.tools import tool

from app.tools._client import flux_api, get_workspace_id, get_project_id
from app.tools._guards import confirm_gate

# MongoDB ObjectId: 24-char hex string
_OID_RE = re.compile(r'^[0-9a-fA-F]{24}$')
_PLACEHOLDER_VALUES = {
    "", "none", "null", "undefined",
    "project_id_not_specified", "not_specified", "not specified",
    "task_id_not_specified",
}


def _is_valid_oid(value: str) -> bool:
    """Return True only if value looks like a real MongoDB ObjectId."""
    return bool(value and _OID_RE.match(value) and value.lower() not in _PLACEHOLDER_VALUES)

def _slim_task(task: dict, column_name: str = "") -> dict:
    """Strip heavy/unused fields from a task object to reduce LLM token usage.

    Only keeps the fields the agent actually needs to reason about tasks.
    Removes: content, description, checklists, attachments, permissions,
             createdAt, updatedAt, rank, reminder, recurrence, estimate, labels.
    """
    assignee = task.get("assignee")
    return {
        "id": str(task.get("_id", "")),
        "identifier": task.get("identifier", ""),
        "title": task.get("title", ""),
        "columnId": task.get("columnId", ""),
        # Human-readable column name — critical so LLM doesn't miss tasks
        "columnName": column_name or task.get("columnId", ""),
        "priority": task.get("priority", "none"),
        "completed": task.get("completed", False),
        "isOverdue": task.get("isOverdue", False),
        "dueState": task.get("dueState", "none"),
        "dueDate": task.get("dueDate"),
        "startDate": task.get("startDate"),
        "assignee": {"id": str(assignee.get("_id", "")), "name": assignee.get("name", "")} if assignee else None,
        "commentCount": task.get("commentCount", 0),
        "cycle": task.get("cycle"),
        "parentTask": task.get("parentTask"),
    }


@tool
async def list_tasks(project_id: str) -> dict:
    """List ALL tasks in a project, grouped by columns (backlog, todo, doing, review, done).

    Use this to get a complete overview of the project's task board.
    Returns EVERY task regardless of who created or is assigned to them.

    IMPORTANT: project_id must be a real 24-char hex MongoDB ObjectId.
    If you don't have it, call list_projects first.

    Args:
        project_id: MongoDB ObjectId of the project.
    """
    # Fallback to context project if not provided
    if not project_id or project_id.lower() in _PLACEHOLDER_VALUES:
        project_id = get_project_id()

    if not _is_valid_oid(project_id):
        return {
            "error": True,
            "message": (
                f"Invalid project_id '{project_id}'. "
                "Please call list_projects first to get valid project IDs."
            ),
        }
    result = await flux_api.get(f"/api/project/{project_id}/tasks")

    if isinstance(result, dict) and result.get("error"):
        return result

    tasks_raw = result.get("tasks", []) if isinstance(result, dict) else []
    columns_raw = result.get("columns", []) if isinstance(result, dict) else []

    # Build column_id → human-readable name map so LLM understands the board
    col_name_map: dict[str, str] = {}
    for col in columns_raw:
        cid = str(col.get("id", col.get("_id", "")))
        ctitle = col.get("title", cid)
        if cid:
            col_name_map[cid] = ctitle

    # Slim tasks — enrich with column name
    slimmed = [_slim_task(t, col_name_map.get(str(t.get("columnId", "")), "")) for t in tasks_raw]

    # Group by column preserving board order
    grouped: dict[str, list] = {}
    col_order = [str(c.get("id", c.get("_id", ""))) for c in columns_raw]

    for cid in col_order:
        grouped[col_name_map.get(cid, cid)] = []

    for t in slimmed:
        cid = str(t.get("columnId", ""))
        cname = col_name_map.get(cid, cid or "Uncategorised")
        if cname not in grouped:
            grouped[cname] = []
        grouped[cname].append(t)

    return {
        "projectName": result.get("projectName", ""),
        "total": len(slimmed),
        "columns": grouped,          # {column_name: [task, ...]} — LLM-friendly
        "tasks": slimmed,            # flat list for tools that need it
    }


@tool
async def get_my_tasks(workspace_id: str = "") -> dict:
    """Get all tasks assigned to the currently logged-in user across ALL projects.

    Use this when the user asks:
    - "task của tôi là gì?"
    - "tôi đang làm gì?"
    - "tóm tắt công việc của tôi"
    - "show my tasks"

    NOTE: This returns ONLY tasks where the current user is the assignee.
    Tasks created without explicitly assigning them will be auto-assigned to the
    creator (so they WILL appear here). Use list_tasks(project_id) to see ALL
    tasks in a project regardless of assignee.

    Returns tasks with project info, status (columnId), priority, dueDate.

    Args:
        workspace_id: Workspace ID or URL slug. If empty, uses the current session workspace.
    """
    ws = workspace_id or get_workspace_id()
    if not ws:
        return {"error": True, "message": "workspace_id is required"}
    result = await flux_api.get(f"/api/workspace/{ws}/tasks")
    if isinstance(result, dict) and not result.get("error"):
        tasks = result.get("tasks", [])
        result = {
            "tasks": [_slim_task(t) for t in tasks],
            "total": len(tasks),
        }
    return result


@tool
async def summarize_member_tasks(project_id: str, member_name: str) -> dict:
    """Get and summarize tasks assigned to a specific team member in a project.

    Use this when the user asks:
    - "task của [tên người] trong project này"
    - "Nguyễn Văn A đang làm gì?"
    - "tóm tắt công việc của [member]"

    Fetches all project tasks then filters by member name (case-insensitive, partial match).

    Args:
        project_id: MongoDB ObjectId of the project.
        member_name: Name (full or partial) of the team member to filter by.
    """
    if not project_id or project_id.lower() in _PLACEHOLDER_VALUES:
        project_id = get_project_id()

    if not _is_valid_oid(project_id):
        return {
            "error": True,
            "message": f"Invalid project_id '{project_id}'. Call list_projects first.",
        }
    result = await flux_api.get(f"/api/project/{project_id}/tasks")
    if isinstance(result, dict) and result.get("error"):
        return result

    tasks = result.get("tasks", []) if isinstance(result, dict) else []
    name_lower = member_name.lower()

    # Filter tasks where assignee name matches
    member_tasks = [
        t for t in tasks
        if t.get("assignee") and
        name_lower in (t["assignee"].get("name", "") or "").lower()
    ]

    if not member_tasks:
        return {
            "member": member_name,
            "found": 0,
            "message": f"No tasks found assigned to '{member_name}' in this project.",
        }

    # Group by status column
    by_status: dict[str, list] = {}
    by_priority: dict[str, int] = {}
    overdue: list = []

    for t in member_tasks:
        col = t.get("columnId", "unknown")
        by_status.setdefault(col, []).append({
            "title": t.get("title"),
            "id": str(t.get("_id", "")),
            "priority": t.get("priority", "none"),
            "dueDate": t.get("dueDate"),
        })

        pri = t.get("priority", "none")
        by_priority[pri] = by_priority.get(pri, 0) + 1

        if t.get("isOverdue") and col != "done":
            overdue.append(t.get("title"))

    return {
        "member": member_name,
        "found": len(member_tasks),
        "by_status": by_status,
        "by_priority": by_priority,
        "overdue_tasks": overdue,
        "project": result.get("projectName", project_id),
    }


@tool
async def create_task(
    project_id: str,
    title: str,
    description: str = "",
    priority: str = "none",
    assignee_id: str | None = None,
    due_date: str | None = None,
    start_date: str | None = None,
    column_id: str = "todo",
    labels: list[str] | None = None,
    estimate: int | None = None,
    cycle_id: str | None = None,
) -> dict:
    """Create a new task in a project.

    IMPORTANT: If no assignee_id is provided, the task is automatically
    assigned to the current user. After creating, use list_tasks(project_id)
    to verify the result — NOT get_my_tasks (which may have different scope).

    Args:
        project_id: MongoDB ObjectId of the project.
        title: Task title (required, descriptive).
        description: Task description (supports markdown).
        priority: One of "urgent", "high", "medium", "low", "none".
        assignee_id: MongoDB ObjectId of the user to assign. If omitted,
                     the task is assigned to the current user automatically.
        due_date: ISO date string (YYYY-MM-DD) for the deadline.
        start_date: ISO date string (YYYY-MM-DD) for the start date.
        column_id: Board column — "backlog", "todo", "in_progress", or "done".
        labels: List of label strings to tag the task.
        estimate: Estimated effort in hours.
        cycle_id: MongoDB ObjectId of a research cycle to attach to.
    """
    body: dict = {
        "title": title,
        "columnId": column_id,
    }
    if not _is_valid_oid(project_id):
        return {
            "error": True,
            "message": f"Invalid project_id '{project_id}'. Call list_projects first to get a valid ID.",
        }
    if description:
        body["description"] = description
    if priority != "none":
        body["priority"] = priority
    if assignee_id:
        body["assignee"] = assignee_id
    if due_date:
        body["dueDate"] = due_date
    if start_date:
        body["startDate"] = start_date
    if labels:
        body["labels"] = labels
    if estimate is not None:
        body["estimate"] = estimate
    if cycle_id:
        body["cycle"] = cycle_id

    return await flux_api.post(f"/api/project/{project_id}/tasks", json=body)


@tool
async def update_task(
    task_id: str,
    title: str | None = None,
    description: str | None = None,
    priority: str | None = None,
    assignee_id: str | None = None,
    due_date: str | None = None,
    start_date: str | None = None,
    column_id: str | None = None,
    labels: list[str] | None = None,
    completed: bool | None = None,
    estimate: int | None = None,
    cycle_id: str | None = None,
) -> dict:
    """Update an existing task. Only provided fields are modified.

    ⚠️ SENSITIVE: This modifies existing data. Confirm with the user if unsure.

    Args:
        task_id: MongoDB ObjectId of the task to update.
        title: New title for the task.
        description: New description (markdown).
        priority: One of "urgent", "high", "medium", "low", "none".
        assignee_id: User ID to re-assign to, or empty string to unassign.
        due_date: ISO date string for the new deadline.
        start_date: ISO date string for the new start date.
        column_id: Move to column — "backlog", "todo", "in_progress", "done".
        labels: Replace all labels with this list.
        completed: Mark task as complete (true) or incomplete (false).
        estimate: New estimated effort in hours.
        cycle_id: Attach to / detach from a research cycle.
    """
    body: dict = {}
    if title is not None:
        body["title"] = title
    if description is not None:
        body["description"] = description
    if priority is not None:
        body["priority"] = priority
    if assignee_id is not None:
        body["assignee"] = assignee_id if assignee_id else None
    if due_date is not None:
        body["dueDate"] = due_date
    if start_date is not None:
        body["startDate"] = start_date
    if column_id is not None:
        body["columnId"] = column_id
    if labels is not None:
        body["labels"] = labels
    if completed is not None:
        body["completed"] = completed
    if estimate is not None:
        body["estimate"] = estimate
    if cycle_id is not None:
        body["cycle"] = cycle_id if cycle_id else None

    if not body:
        return {"error": True, "message": "No fields to update"}

    return await flux_api.put(f"/api/tasks/{task_id}", json=body)


@tool
async def delete_task(task_id: str) -> dict:
    """Delete a task permanently.

    ⚠️ DESTRUCTIVE: This action cannot be undone. Always confirm with the user before deleting.

    Args:
        task_id: MongoDB ObjectId of the task to delete.
    """
    block = confirm_gate(f"delete task `{task_id}`")
    if block:
        return block
    return await flux_api.delete(f"/api/tasks/{task_id}")
