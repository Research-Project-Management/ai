"""
Project Sub-Agent — Handles project & member management.

Invoked by Orchestrator when the user wants to:
- List/view projects
- Get project overview/progress
- Manage project members (add, remove, update role)
- Update project settings
"""

from __future__ import annotations

from app.agents.card import AgentCard, AgentTask, AgentResult, register_card  # noqa: F401
from app.agents.sub_agents.base_sub_agent import BaseSubAgent
from app.tools.project.project_tools import (
    list_projects,
    get_project_overview,
    get_project_details,
    update_project,
    add_project_member,
    remove_project_member,
    update_member_role,
)
from app.tools.workspace.workspace_tools import list_members

# ── Agent Card ─────────────────────────────────────────────────────────────

PROJECT_AGENT_CARD = register_card(AgentCard(
    name="project_agent",
    description=(
        "Manages projects and team members: listing projects, viewing project overview, "
        "progress reports, adding/removing members, updating project settings."
    ),
    skills=[
        "list_projects", "get_project_overview", "get_project_details",
        "update_project", "add_project_member", "remove_project_member",
        "update_member_role", "list_members",
    ],
    has_destructive_ops=True,
))

# ── System Prompt ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are the **Project Agent** — a specialized AI for managing projects and team \
members in the user's research workspace.

## Your Scope
You handle: project listing, project overview/progress, member management, \
and project settings. You do NOT handle task CRUD (the Task Agent handles that).

## Context
- Workspace ID: {workspace_id}
- Active Project ID: {project_id}
- User ID: {user_id}

## Rules
1. For member operations (add/remove/update role), always confirm the member's \
   user_id by calling list_members first.
2. For progress reports, use get_project_overview for stats.
3. Always respond in the same language as the user.
4. Use markdown tables for member lists and progress summaries.
"""


# ── Sub-Agent ──────────────────────────────────────────────────────────────

class ProjectSubAgent(BaseSubAgent):
    """Project & member management sub-agent."""

    card = PROJECT_AGENT_CARD
    tools = [
        list_projects, get_project_overview, get_project_details,
        update_project, add_project_member, remove_project_member,
        update_member_role, list_members,
    ]

    def _build_prompt(self, task: AgentTask) -> str:
        return _SYSTEM_PROMPT.format(
            workspace_id=task.workspace_id or "N/A",
            project_id=task.project_id or "none",
            user_id=task.user_id or "unknown",
        )
