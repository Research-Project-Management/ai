"""Project management tools — read project info and overviews."""

from __future__ import annotations

from langchain_core.tools import tool

from app.tools._client import flux_api, get_workspace_id


@tool
async def list_projects(workspace_id: str = "") -> dict:
    """List all projects in a workspace.

    Use this to find project IDs before performing task or page operations.

    Args:
        workspace_id: Workspace ID or URL slug. Leave empty to use the current workspace context.
    """
    ws_id = workspace_id or get_workspace_id()
    if not ws_id:
        return {"error": True, "message": "No workspace_id available"}
    return await flux_api.get(f"/api/workspace/{ws_id}/projects")


@tool
async def get_project_overview(project_id: str) -> dict:
    """Get a detailed project overview including task statistics, recent activity, and member list.

    This returns rich data: task counts by status, recent tasks, member list, and cycles.
    Use this when the user asks about project progress, status summaries, or team workload.

    Args:
        project_id: MongoDB ObjectId of the project.
    """
    return await flux_api.get(f"/api/project/{project_id}/overview")
