"""Page management tools — list, create, update, delete research pages."""

from __future__ import annotations

from langchain_core.tools import tool

from app.tools._client import flux_api, get_workspace_id


@tool
async def list_pages(project_id: str, search: str = "") -> dict:
    """List all top-level pages in a project.

    Use this to see what documents exist in the project.

    Args:
        project_id: MongoDB ObjectId of the project.
        search: Optional search keyword to filter pages by title.
    """
    params = {}
    if search:
        params["search"] = search
    return await flux_api.get(f"/api/project/{project_id}/pages", params=params)


@tool
async def get_page_content(page_id: str) -> dict:
    """Get the full content and metadata of a specific page.

    Use this to read the content of a LaTeX document or research page.

    Args:
        page_id: MongoDB ObjectId of the page.
    """
    return await flux_api.get(f"/api/pages/{page_id}")


@tool
async def create_page(
    project_id: str,
    title: str,
    content: str = "",
    status: str = "draft",
) -> dict:
    """Create a new page (LaTeX document) in a project.

    This creates a root page AND a main.tex child file automatically.
    Use this when the user wants to start writing a new document or report.

    Args:
        project_id: MongoDB ObjectId of the project.
        title: Page title (becomes the LaTeX project name).
        content: Initial LaTeX content for main.tex (optional).
        status: One of "draft", "published", "archived".
    """
    body: dict = {"title": title, "status": status}
    if content:
        body["content"] = content
    return await flux_api.post(f"/api/project/{project_id}/pages", json=body)


@tool
async def update_page(
    page_id: str,
    title: str | None = None,
    status: str | None = None,
) -> dict:
    """Update page metadata (title or status). Does NOT update LaTeX content.

    Use this to rename a page or change its status.

    Args:
        page_id: MongoDB ObjectId of the page.
        title: New title for the page.
        status: One of "draft", "published", "archived".
    """
    body: dict = {}
    if title is not None:
        body["title"] = title
    if status is not None:
        body["status"] = status

    if not body:
        return {"error": True, "message": "No fields to update"}

    return await flux_api.put(f"/api/pages/{page_id}", json=body)


@tool
async def delete_page(page_id: str) -> dict:
    """Delete a page and all its child files permanently.

    ⚠️ DESTRUCTIVE: This deletes the entire LaTeX project folder. Always confirm with user.

    Args:
        page_id: MongoDB ObjectId of the page to delete.
    """
    return await flux_api.delete(f"/api/pages/{page_id}")


@tool
async def get_page_versions(page_id: str) -> dict:
    """Get version history of a page (manual saves only).

    Use this to see who saved which versions and when.

    Args:
        page_id: MongoDB ObjectId of the page.
    """
    return await flux_api.get(f"/api/pages/{page_id}/versions")


@tool
async def list_workspace_pages(workspace_id: str = "", search: str = "") -> dict:
    """List all pages across every project in the workspace.

    Use this when the user wants to find a document but doesn't know which project it's in.

    Args:
        workspace_id: Workspace ID. If empty, uses current session workspace.
        search: Optional keyword to filter by title.
    """
    ws = workspace_id or get_workspace_id()
    if not ws:
        return {"error": True, "message": "workspace_id is required"}
    params = {}
    if search:
        params["search"] = search
    return await flux_api.get(f"/api/workspace/{ws}/pages", params=params)
