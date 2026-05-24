"""
Workspace Sub-Agent — Handles workspace-level operations & analysis.

Invoked by Orchestrator when the user wants:
- Workspace overview (recent items, all projects summary)
- Activity feed (who did what recently)
- Global search across workspace
- Stickies, tags, page listing
- Analysis: workload distribution, overdue report, velocity, team summary
"""

from __future__ import annotations

from app.agents.card import AgentCard, AgentTask, AgentResult, register_card  # noqa: F401
from app.agents.sub_agents.base_sub_agent import BaseSubAgent
from app.tools.workspace.workspace_tools import (
    search_workspace,
    list_members,
    get_workspace_overview,
    get_workspace_activity,
)
from app.tools.workspace.sticky_tools import list_stickies, create_sticky, update_sticky, delete_sticky
from app.tools.workspace.tag_tools import list_tags, create_tag, delete_tag
from app.tools.project.page_tools import list_pages, list_workspace_pages, get_page_content
from app.tools.workspace.user_tools import get_current_user, search_users
from app.tools.workspace.analysis_tools import (
    get_workload_distribution,
    get_overdue_report,
    get_project_velocity,
    generate_team_summary,
)

# ── Agent Card ─────────────────────────────────────────────────────────────

WORKSPACE_AGENT_CARD = register_card(AgentCard(
    name="workspace_agent",
    description=(
        "Handles workspace-level queries: overview (all projects, recent activity), "
        "global search, sticky notes, tags, workspace members, and analytical reports "
        "(workload distribution, overdue report, team velocity)."
    ),
    skills=[
        "get_workspace_overview", "get_workspace_activity", "search_workspace",
        "list_members", "list_stickies", "create_sticky",
        "get_workload_distribution", "get_overdue_report", "generate_team_summary",
    ],
    has_destructive_ops=False,
))

# ── System Prompt ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are the **Workspace Agent** — a highly analytical AI specialized in workspace-level insights, administrative operations, and project portfolio analysis in Flux-AI.

## Your Scope
- Workspace overviews, activity feeds, global search
- Sticky notes management
- Tags management
- Analytical reports (workload distribution, overdue task audits, project velocity, team summaries)
- User lookup

You do NOT handle task CRUD or project member management (other specialized agents handle those).

## Context
- Workspace ID: {workspace_id}
- User ID: {user_id}

## ANALYTICAL QUALITY PROTOCOL & STYLE
1. **Strategic Perspective**: When compiling reports (workload, velocity, team summary), do not just dump raw data. Accompany every table or summary with a strategic, high-level analysis. Point out potential operational bottlenecks (e.g., "User X has 70% of high-priority tasks assigned, indicating a potential single-point-of-failure risk"), team velocity implications, and overdue risk vectors.
2. **Visual & Typographical Class**: Format all analytics using beautifully structured Markdown tables with descriptive, professional headers (e.g., metric, status, assignee, risk index). Use nested lists for secondary details.
3. **Bilingual Excellence**: Always respond in the same language as the user's request. When writing in Vietnamese, use natural, highly polished, and authoritative professional terms (e.g., 'tốc độ dự án', 'phân bổ nguồn lực', 'rủi ro nghẽn cổ chai').

## Rules
1. For "workspace có gì?" or general status queries → utilize get_workspace_overview first to analyze the project status.
2. For "có gì mới?" or recent activity → use get_workspace_activity to present a unified chronological feed.
3. For analysis requests → use the dedicated analytical tools (`get_workload_distribution`, `get_overdue_report`, `get_project_velocity`, `generate_team_summary`).
"""


# ── Sub-Agent ──────────────────────────────────────────────────────────────

class WorkspaceSubAgent(BaseSubAgent):
    """Workspace-level operations and analytics sub-agent."""

    card = WORKSPACE_AGENT_CARD
    tools = [
        search_workspace, list_members, get_workspace_overview, get_workspace_activity,
        list_stickies, create_sticky, update_sticky, delete_sticky,
        list_tags, create_tag, delete_tag,
        list_workspace_pages, list_pages, get_page_content,
        get_current_user, search_users,
        get_workload_distribution, get_overdue_report, get_project_velocity, generate_team_summary,
    ]

    def _build_prompt(self, task: AgentTask) -> str:
        return _SYSTEM_PROMPT.format(
            workspace_id=task.workspace_id or "N/A",
            user_id=task.user_id or "unknown",
        )
