"""Comment tools — read and add comments on tasks and pages."""

from __future__ import annotations

from langchain_core.tools import tool

from app.tools._client import flux_api


@tool
async def get_task_comments(task_id: str) -> dict:
    """Get all comments on a specific task, including replies.

    Use this to see what the team is discussing about a task, check for
    blockers, decisions, or context buried in the comment thread.

    Args:
        task_id: MongoDB ObjectId of the task.
    """
    return await flux_api.get(f"/api/tasks/{task_id}/comments")


@tool
async def add_task_comment(task_id: str, content: str) -> dict:
    """Add a comment to a task on behalf of the user.

    Use this to:
    - Leave a note about task progress or blockers
    - Document a decision related to this task
    - Notify teammates by adding context

    Args:
        task_id: MongoDB ObjectId of the task.
        content: Comment text (markdown supported).
    """
    return await flux_api.post(f"/api/tasks/{task_id}/comments", json={"content": content})


@tool
async def get_page_comments(page_id: str) -> dict:
    """Get all comments on a specific page/document.

    Use this to see review notes, feedback, and discussions on a research document.

    Args:
        page_id: MongoDB ObjectId of the page.
    """
    return await flux_api.get(f"/api/pages/{page_id}/comments")


@tool
async def add_page_comment(page_id: str, content: str) -> dict:
    """Add a review comment to a page/document.

    Use this to:
    - Leave feedback on a research document
    - Flag sections for revision
    - Add general notes to a page

    Args:
        page_id: MongoDB ObjectId of the page.
        content: Comment text (markdown supported).
    """
    return await flux_api.post(f"/api/pages/{page_id}/comments", json={"content": content})
