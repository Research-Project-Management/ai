"""Workspace tools — member lookup, search, home data."""

from __future__ import annotations

from langchain_core.tools import tool

from app.tools._client import flux_api, get_workspace_id


@tool
async def list_members(workspace_id: str = "") -> dict:
    """List all members of a workspace with their names, IDs, and roles.

    IMPORTANT: Always call this before assigning tasks to find the correct user ID.
    Returns member list with: _id, name, email, avatar, role.

    Args:
        workspace_id: Workspace ID or URL slug. Leave empty to use current context.
    """
    ws_id = workspace_id or get_workspace_id()
    if not ws_id:
        return {"error": True, "message": "No workspace_id available"}

    data = await flux_api.get(f"/api/workspace/{ws_id}")

    # Extract just the members from the workspace data
    if isinstance(data, dict) and "workspace" in data:
        ws = data["workspace"]
        members = ws.get("members", [])
        return {
            "members": [
                {
                    "user_id": str(m.get("user", {}).get("_id", m.get("user", ""))),
                    "name": m.get("user", {}).get("name", "Unknown"),
                    "email": m.get("user", {}).get("email", ""),
                    "avatar": m.get("user", {}).get("avatar", ""),
                    "role": m.get("role", {}).get("name", "member"),
                }
                for m in members
                if isinstance(m, dict)
            ]
        }
    return data


@tool
async def search_workspace(query: str, workspace_id: str = "") -> dict:
    """Search across all content in a workspace — tasks, pages, projects, and members.

    Use this when the user asks to find something by keyword or when you need
    to locate a specific item before performing an action.

    Args:
        query: Search query string.
        workspace_id: Workspace ID or URL slug. Leave empty to use current context.
    """
    ws_id = workspace_id or get_workspace_id()
    if not ws_id:
        return {"error": True, "message": "No workspace_id available"}
    return await flux_api.get(
        f"/api/workspace/{ws_id}/search",
        params={"q": query},
    )


@tool
async def get_workspace_overview(workspace_id: str = "") -> dict:
    """Get a comprehensive overview of the current workspace — its name, members,
    recent pages, recent projects, and recent files.

    Use this when the user asks:
    - "workspace của mình có gì?"
    - "show me workspace info"
    - "what projects do we have?"
    - Any general question about the workspace contents.

    Args:
        workspace_id: Workspace ID or URL slug. Leave empty to use current context.
    """
    ws_id = workspace_id or get_workspace_id()
    if not ws_id:
        return {"error": True, "message": "No workspace_id available"}

    # Fetch workspace info and recent items in parallel
    import asyncio

    ws_info, recent = await asyncio.gather(
        flux_api.get(f"/api/workspace/{ws_id}"),
        flux_api.get(f"/api/workspace/{ws_id}/recent"),
        return_exceptions=True,
    )

    result: dict = {}

    # Workspace basic info
    if isinstance(ws_info, dict) and "workspace" in ws_info:
        ws = ws_info["workspace"]
        result["workspace"] = {
            "name": ws.get("name", ""),
            "url": ws.get("url", ""),
            "avatar": ws.get("avatar", ""),
            "member_count": len(ws.get("members", [])),
        }

    # Recent items
    if isinstance(recent, dict) and "items" in recent:
        result["recent_items"] = recent["items"]
    elif isinstance(recent, Exception):
        result["recent_items"] = []

    return result


@tool
async def get_workspace_activity(workspace_id: str = "") -> dict:
    """Get the recent activity feed of a workspace — who did what recently.

    Use this when the user asks:
    - "có gì mới không?"
    - "what happened recently?"
    - "show recent activity"

    Args:
        workspace_id: Workspace ID or URL slug. Leave empty to use current context.
    """
    ws_id = workspace_id or get_workspace_id()
    if not ws_id:
        return {"error": True, "message": "No workspace_id available"}

    return await flux_api.get(f"/api/workspace/{ws_id}/activity")

