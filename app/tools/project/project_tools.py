"""Project management tools — update projects, manage members."""

from __future__ import annotations

from langchain_core.tools import tool

from app.tools._client import flux_api, get_workspace_id


@tool
async def list_projects(workspace_id: str = "") -> dict:
    """List all projects in the workspace that the user has access to.

    Use this to get project names and IDs before performing project-level operations.

    Args:
        workspace_id: Workspace ID or URL slug. If empty, uses current session workspace.
    """
    ws = workspace_id or get_workspace_id()
    if not ws:
        return {"error": True, "message": "workspace_id is required"}
    return await flux_api.get(f"/api/workspace/{ws}/projects")


@tool
async def get_project_overview(project_id: str) -> dict:
    """Get the overview of a project including file stats and member count.

    Use this for a high-level summary of what a project contains.

    Args:
        project_id: MongoDB ObjectId of the project.
    """
    return await flux_api.get(f"/api/project/{project_id}/overview")


@tool
async def get_project_details(project_id: str) -> dict:
    """Get detailed information about a project including all members and their roles.

    Use this to see who is in a project, their roles, and project settings.

    Args:
        project_id: MongoDB ObjectId of the project.
    """
    return await flux_api.get(f"/api/project/{project_id}")


@tool
async def update_project(
    project_id: str,
    name: str | None = None,
    description: str | None = None,
    avatar: str | None = None,
    is_active: bool | None = None,
) -> dict:
    """Update project metadata such as name, description, or avatar emoji.

    ⚠️ SENSITIVE: Requires manager role on the project.

    Args:
        project_id: MongoDB ObjectId of the project.
        name: New project name.
        description: New project description.
        avatar: Emoji or icon for the project.
        is_active: Set to false to archive/deactivate the project.
    """
    body: dict = {}
    if name is not None:
        body["name"] = name
    if description is not None:
        body["description"] = description
    if avatar is not None:
        body["avatar"] = avatar
    if is_active is not None:
        body["isActive"] = is_active

    if not body:
        return {"error": True, "message": "No fields to update"}

    return await flux_api.put(f"/api/project/{project_id}", json=body)


@tool
async def add_project_member(
    project_id: str,
    user_id: str,
    role: str = "member",
) -> dict:
    """Add a user to a project with a specific role.

    Use this when the user wants to invite someone to a project.
    The user must already be a member of the workspace.
    Use search_users() first to find the user's ID.

    Args:
        project_id: MongoDB ObjectId of the project.
        user_id: MongoDB ObjectId of the user to add.
        role: Role to assign — one of "manager", "member", "viewer".
    """
    return await flux_api.put(
        f"/api/project/{project_id}/add-member",
        json={"userId": user_id, "role": role},
    )


@tool
async def remove_project_member(project_id: str, user_id: str) -> dict:
    """Remove a user from a project.

    ⚠️ SENSITIVE: The user will lose access to all project content. Confirm with user.

    Args:
        project_id: MongoDB ObjectId of the project.
        user_id: MongoDB ObjectId of the user to remove.
    """
    return await flux_api.put(
        f"/api/project/{project_id}/remove-member",
        json={"userId": user_id},
    )


@tool
async def update_member_role(
    project_id: str,
    user_id: str,
    new_role: str,
) -> dict:
    """Change a project member's role.

    ⚠️ SENSITIVE: This changes the user's permissions in the project.

    Args:
        project_id: MongoDB ObjectId of the project.
        user_id: MongoDB ObjectId of the user.
        new_role: New role — one of "manager", "member", "viewer".
    """
    return await flux_api.put(
        f"/api/project/{project_id}/update-member-role",
        json={"userId": user_id, "newRole": new_role},
    )
