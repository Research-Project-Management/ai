"""Sticky notes tools — create, update, delete workspace sticky notes."""

from __future__ import annotations

from langchain_core.tools import tool

from app.tools._client import flux_api, get_workspace_id


@tool
async def list_stickies(workspace_id: str = "") -> dict:
    """List all sticky notes in the workspace.

    Use this to see quick notes and reminders the team has left.

    Args:
        workspace_id: Workspace ID. If empty, uses current session workspace.
    """
    ws = workspace_id or get_workspace_id()
    if not ws:
        return {"error": True, "message": "workspace_id is required"}
    return await flux_api.get(f"/api/workspace/{ws}/stickies")


@tool
async def create_sticky(
    content: str,
    title: str = "",
    color: str = "yellow-1",
    workspace_id: str = "",
) -> dict:
    """Create a new sticky note in the workspace.

    Use this to leave quick notes, reminders, or ideas for the team.

    Args:
        content: Main content of the sticky note (markdown supported).
        title: Optional title for the sticky note.
        color: Color theme — one of "yellow-1", "blue-1", "green-1", "pink-1", "purple-1".
        workspace_id: Workspace ID. If empty, uses current session workspace.
    """
    ws = workspace_id or get_workspace_id()
    if not ws:
        return {"error": True, "message": "workspace_id is required"}
    body: dict = {"content": content, "color": color}
    if title:
        body["title"] = title
    return await flux_api.post(f"/api/workspace/{ws}/stickies", json=body)


@tool
async def update_sticky(
    sticky_id: str,
    content: str | None = None,
    title: str | None = None,
    color: str | None = None,
) -> dict:
    """Update an existing sticky note's content, title, or color.

    Args:
        sticky_id: MongoDB ObjectId of the sticky note.
        content: New main content.
        title: New title.
        color: New color — one of "yellow-1", "blue-1", "green-1", "pink-1", "purple-1".
    """
    body: dict = {}
    if content is not None:
        body["content"] = content
    if title is not None:
        body["title"] = title
    if color is not None:
        body["color"] = color

    if not body:
        return {"error": True, "message": "No fields to update"}

    return await flux_api.put(f"/api/stickies/{sticky_id}", json=body)


@tool
async def delete_sticky(sticky_id: str) -> dict:
    """Delete a sticky note permanently.

    ⚠️ DESTRUCTIVE: This cannot be undone. Always confirm with the user first.

    Args:
        sticky_id: MongoDB ObjectId of the sticky note to delete.
    """
    return await flux_api.delete(f"/api/stickies/{sticky_id}")
