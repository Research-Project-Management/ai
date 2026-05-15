"""Cycle management tools — create, read, update, delete research cycles."""

from __future__ import annotations

from langchain_core.tools import tool

from app.tools._client import flux_api
from app.tools._guards import confirm_gate


@tool
async def list_cycles(project_id: str) -> dict:
    """List all research cycles in a project, including task progress stats.

    Use this to see what phases of research exist, their status, and progress.

    Args:
        project_id: MongoDB ObjectId of the project.
    """
    return await flux_api.get(f"/api/project/{project_id}/cycles")


@tool
async def get_cycle_details(project_id: str, cycle_name: str) -> dict:
    """Get detailed information about a specific cycle by name.

    Use this when the user asks about a specific research cycle's tasks or progress.

    Args:
        project_id: MongoDB ObjectId of the project.
        cycle_name: Name (full or partial) of the cycle to look up.
    """
    result = await flux_api.get(f"/api/project/{project_id}/cycles")
    if isinstance(result, dict) and result.get("error"):
        return result

    cycles = result.get("cycles", []) if isinstance(result, dict) else []
    name_lower = cycle_name.lower()

    matched = [c for c in cycles if name_lower in (c.get("name", "") or "").lower()]
    if not matched:
        return {"found": 0, "message": f"No cycle found matching '{cycle_name}'"}

    return {"found": len(matched), "cycles": matched}


@tool
async def create_cycle(
    project_id: str,
    name: str,
    phase: str = "custom",
    description: str = "",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Create a new research cycle in a project.

    Use this to plan a new research phase or sprint.

    Args:
        project_id: MongoDB ObjectId of the project.
        name: Cycle name (e.g., "Literature Review", "Experiment Phase 1").
        phase: One of "planning", "research", "writing", "review", "custom".
        description: Optional description of the cycle's goals.
        start_date: ISO date string (YYYY-MM-DD).
        end_date: ISO date string (YYYY-MM-DD).
    """
    body: dict = {"name": name, "phase": phase}
    if description:
        body["description"] = description
    if start_date:
        body["startDate"] = start_date
    if end_date:
        body["endDate"] = end_date
    return await flux_api.post(f"/api/project/{project_id}/cycles", json=body)


@tool
async def update_cycle(
    cycle_id: str,
    name: str | None = None,
    description: str | None = None,
    phase: str | None = None,
    status: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Update an existing research cycle.

    ⚠️ SENSITIVE: Modifies existing data.

    Args:
        cycle_id: MongoDB ObjectId of the cycle to update.
        name: New cycle name.
        description: New description.
        phase: One of "planning", "research", "writing", "review", "custom".
        status: One of "active", "completed", "paused".
        start_date: ISO date string (YYYY-MM-DD).
        end_date: ISO date string (YYYY-MM-DD).
    """
    body: dict = {}
    if name is not None:
        body["name"] = name
    if description is not None:
        body["description"] = description
    if phase is not None:
        body["phase"] = phase
    if status is not None:
        body["status"] = status
    if start_date is not None:
        body["startDate"] = start_date
    if end_date is not None:
        body["endDate"] = end_date

    if not body:
        return {"error": True, "message": "No fields to update"}

    return await flux_api.put(f"/api/cycles/{cycle_id}", json=body)


@tool
async def delete_cycle(cycle_id: str) -> dict:
    """Delete a research cycle permanently.

    ⚠️ DESTRUCTIVE: This will unlink all tasks from the cycle. Always confirm with user.

    Args:
        cycle_id: MongoDB ObjectId of the cycle to delete.
    """
    block = confirm_gate(f"delete cycle `{cycle_id}`")
    if block:
        return block
    return await flux_api.delete(f"/api/cycles/{cycle_id}")
