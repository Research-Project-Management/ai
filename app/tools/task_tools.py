"""Task management tools — CRUD operations on project tasks."""

from __future__ import annotations

from langchain_core.tools import tool

from app.tools._client import flux_api


@tool
async def list_tasks(project_id: str) -> dict:
    """List all tasks in a project, grouped by columns (backlog, todo, in_progress, done).

    Use this to get an overview of the project's task board, check what tasks exist,
    or gather data before creating/updating tasks.

    Args:
        project_id: MongoDB ObjectId of the project.
    """
    return await flux_api.get(f"/api/project/{project_id}/tasks")


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

    Use this to create tasks for team members. Always check list_members first
    if you need to assign a task to someone.

    Args:
        project_id: MongoDB ObjectId of the project.
        title: Task title (required, descriptive).
        description: Task description (supports markdown).
        priority: One of "urgent", "high", "medium", "low", "none".
        assignee_id: MongoDB ObjectId of the user to assign. Get from list_members.
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
    return await flux_api.delete(f"/api/tasks/{task_id}")
