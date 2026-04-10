"""Page tools — read-only page access for Phase 1."""

from __future__ import annotations

from langchain_core.tools import tool

from app.tools._client import flux_api


@tool
async def list_pages(project_id: str) -> dict:
    """List all pages in a project.

    Returns page titles, IDs, and metadata. Use this to find pages
    before retrieving their content.

    Args:
        project_id: MongoDB ObjectId of the project.
    """
    return await flux_api.get(f"/api/project/{project_id}/pages")


@tool
async def get_page_content(page_id: str) -> dict:
    """Get the full content of a specific page.

    Returns the page title, content (HTML/markdown), and metadata.
    Useful when the user asks about a specific page or wants a summary.

    Args:
        page_id: MongoDB ObjectId of the page.
    """
    return await flux_api.get(f"/api/pages/{page_id}")
