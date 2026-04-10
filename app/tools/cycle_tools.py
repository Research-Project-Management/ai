"""Research cycle tools — manage research phases and milestones."""

from __future__ import annotations

from langchain_core.tools import tool

from app.tools._client import flux_api


@tool
async def list_cycles(project_id: str) -> dict:
    """List all research cycles in a project.

    Cycles represent research phases (e.g., "Literature Review", "Data Collection").
    Each cycle has tasks attached to it.

    Args:
        project_id: MongoDB ObjectId of the project.
    """
    return await flux_api.get(f"/api/project/{project_id}/cycles")


@tool
async def create_cycle(
    project_id: str,
    name: str,
    description: str = "",
    start_date: str | None = None,
    end_date: str | None = None,
    phase: str = "planning",
) -> dict:
    """Create a new research cycle (phase) in a project.

    Use this to organize tasks into research phases like "Literature Review",
    "Methodology", "Data Collection", "Analysis", "Writing".

    Args:
        project_id: MongoDB ObjectId of the project.
        name: Cycle name (e.g., "Literature Review Q1 2026").
        description: Detailed description of the cycle goals.
        start_date: ISO date string (YYYY-MM-DD) for cycle start.
        end_date: ISO date string (YYYY-MM-DD) for cycle end.
        phase: Current phase — "planning", "active", "review", "completed".
    """
    body: dict = {"name": name}
    if description:
        body["description"] = description
    if start_date:
        body["startDate"] = start_date
    if end_date:
        body["endDate"] = end_date
    if phase:
        body["phase"] = phase

    return await flux_api.post(f"/api/project/{project_id}/cycles", json=body)
