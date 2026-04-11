"""Tag management tools — list, create, delete workspace tags."""

from __future__ import annotations

from langchain_core.tools import tool

from app.tools._client import flux_api, get_workspace_id


@tool
async def list_tags(workspace_id: str = "") -> dict:
    """List all tags in the workspace.

    Tags are workspace-level and can be applied to tasks. Use this before
    creating tasks with labels to see what tags already exist.

    Args:
        workspace_id: Workspace ID or URL slug. If empty, uses current session workspace.
    """
    ws = workspace_id or get_workspace_id()
    if not ws:
        return {"error": True, "message": "workspace_id is required"}
    return await flux_api.get(f"/api/workspace/{ws}/tags")


@tool
async def create_tag(
    name: str,
    color: str = "#3b82f6",
    workspace_id: str = "",
) -> dict:
    """Create a new tag in the workspace.

    Use this when the user wants to add a new label/category for organizing tasks.

    Args:
        name: Tag name (must be unique within workspace).
        color: Hex color code (e.g., "#ef4444" for red, "#22c55e" for green).
        workspace_id: Workspace ID. If empty, uses current session workspace.
    """
    ws = workspace_id or get_workspace_id()
    if not ws:
        return {"error": True, "message": "workspace_id is required"}
    return await flux_api.post(f"/api/workspace/{ws}/tags", json={"name": name, "color": color})


@tool
async def delete_tag(tag_id: str) -> dict:
    """Delete a tag from the workspace.

    ⚠️ DESTRUCTIVE: This removes the tag from all tasks that use it.

    Args:
        tag_id: MongoDB ObjectId of the tag to delete.
    """
    return await flux_api.delete(f"/api/tags/{tag_id}")
