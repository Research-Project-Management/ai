"""
Cycle Sub-Agent — Handles research sprint / cycle operations.

Invoked by Orchestrator when the user wants to:
- List, create, update, delete research cycles
- Get cycle details and progress
"""

from __future__ import annotations

from app.agents.card import AgentCard, AgentTask, AgentResult, register_card  # noqa: F401
from app.agents.sub_agents.base_sub_agent import BaseSubAgent
from app.tools.cycle.cycle_tools import (
    list_cycles,
    get_cycle_details,
    create_cycle,
    update_cycle,
    delete_cycle,
)

# ── Agent Card ─────────────────────────────────────────────────────────────

CYCLE_AGENT_CARD = register_card(AgentCard(
    name="cycle_agent",
    description=(
        "Manages research cycles (sprints): listing cycles, creating new research sprints, "
        "tracking cycle progress, updating cycle status, and deleting cycles."
    ),
    skills=["list_cycles", "get_cycle_details", "create_cycle", "update_cycle", "delete_cycle"],
    has_destructive_ops=True,
))

# ── System Prompt ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are the **Cycle Agent** — a specialized AI for managing research cycles (sprints) \
in the user's workspace.

## Context
- Workspace ID: {workspace_id}
- Active Project ID: {project_id}
- User ID: {user_id}

## Rules
1. Cycles are research sprints with a start date, end date, and phase (planning/active/completed).
2. Always use ISO format for dates (YYYY-MM-DD).
3. For cycle progress, call get_cycle_details.
4. Respond in the same language as the user.
"""


# ── Sub-Agent ──────────────────────────────────────────────────────────────

class CycleSubAgent(BaseSubAgent):
    """Research cycle management sub-agent."""

    card = CYCLE_AGENT_CARD
    tools = [list_cycles, get_cycle_details, create_cycle, update_cycle, delete_cycle]

    def _build_prompt(self, task: AgentTask) -> str:
        return _SYSTEM_PROMPT.format(
            workspace_id=task.workspace_id or "N/A",
            project_id=task.project_id or "none",
            user_id=task.user_id or "unknown",
        )
