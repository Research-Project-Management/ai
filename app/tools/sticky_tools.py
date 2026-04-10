"""Sticky note tools — workspace-level quick notes."""

from __future__ import annotations

from langchain_core.tools import tool

from app.tools._client import flux_api, get_workspace_id


@tool
async def list_stickies(workspace_id: str = "") -> dict:
    """List all sticky notes in a workspace.

    Sticky notes are quick, color-coded workspace-level notes.

    Args:
        workspace_id: Workspace ID or URL slug. Leave empty to use current context.
    """
    ws_id = workspace_id or get_workspace_id()
    if not ws_id:
        return {"error": True, "message": "No workspace_id available"}
    return await flux_api.get(f"/api/workspace/{ws_id}/stickies")


@tool
async def create_sticky(
    title: str = "",
    content: str = "",
    color: str = "yellow-1",
    workspace_id: str = "",
) -> dict:
    """Create a new sticky note in the workspace.

    Use this for quick reminders, notes, or ideas that don't belong to a specific task.

    Args:
        title: Sticky note title (optional).
        content: Sticky note content text.
        color: Color theme — "yellow-1", "blue-1", "green-1", "pink-1", "purple-1".
        workspace_id: Workspace ID or URL slug. Leave empty to use current context.
    """
    ws_id = workspace_id or get_workspace_id()
    if not ws_id:
        return {"error": True, "message": "No workspace_id available"}

    body: dict = {"color": color}
    if title:
        body["title"] = title
    if content:
        body["content"] = content

    return await flux_api.post(f"/api/workspace/{ws_id}/stickies", json=body)
