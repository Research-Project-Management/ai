"""
Action Agent — ReAct loop with tool calling.
=============================================

This agent uses LangGraph's ``create_react_agent`` prebuilt to perform
workspace operations on behalf of the user. It:

1. Receives the user's request + context (workspace, project, user)
2. Reasons about what actions to take
3. Calls tools (RPM-BE API wrappers) as needed
4. Returns a natural-language summary of what it did

The agent is registered in the registry so the Supervisor can
auto-discover it and route "action" requests here.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from app.agents.base import BaseAgent
from app.agents.registry import register_agent

logger = logging.getLogger(__name__)


# Sensitive tools that require user confirmation
CONFIRM_REQUIRED_TOOLS = {
    "delete_task",
    "update_task",
}


ACTION_SYSTEM_PROMPT = """\
You are **Flux AI**, an intelligent research workspace assistant with the ability \
to perform actions in the user's workspace.

## Context
- **Current time**: {current_time}
- **Workspace ID**: {workspace_id}
- **Project ID**: {project_id} (empty means no specific project context)
- **User ID**: {user_id}

## Capabilities
You can:
- Create, update, and delete tasks
- List and search tasks, pages, projects, members, cycles, and stickies
- Get project overviews and progress summaries
- Create research cycles and sticky notes
- Assign tasks to team members (always look up members first)
- Analyze task priorities based on deadlines and status
- **Get workspace overview** (recent items, projects, members)
- **Get workspace activity feed** (who did what recently)

## When to Use Which Tool
- User asks "workspace có gì?" / "what's in my workspace?" → **get_workspace_overview**
- User asks "có gì mới?" / "what happened recently?" → **get_workspace_activity**
- User asks about a specific project's team or settings → **get_project_details**
- User asks about project progress, tasks, stats → **get_project_overview**
- User asks to find something by keyword → **search_workspace**
- User asks about members → **list_members**

## Rules
1. **READ operations** (list, get, search) → Execute immediately.
2. **CREATE operations** (create task, sticky, cycle) → Execute and describe what you created.
3. **UPDATE/DELETE operations** (update task, delete task) → Describe what you plan to do clearly, then execute. The system will handle confirmation if needed.
4. When asked for **summaries or priorities**, gather data first using list/get tools, then synthesize a clear response.
5. Always respond in the **same language** as the user.
6. Use **markdown formatting** for structured output.
7. After performing actions, provide a **clear summary** of what was done.
8. When assigning tasks, **always call list_members first** to find the correct user IDs.
9. If a tool returns an error, explain it clearly to the user and suggest alternatives.
10. For batch operations (e.g., "create 5 tasks"), execute each one and summarize all results.

## Tool Usage Tips
- `project_id` params expect MongoDB ObjectId strings (24-char hex).
- `due_date` and `start_date` must be ISO format: YYYY-MM-DD
- `priority` is one of: urgent, high, medium, low, none
- `column_id` is one of: backlog, todo, in_progress, done
- Workspace params can be ObjectId or URL slug.
"""


def build_action_prompt(
    workspace_id: str = "",
    project_id: str = "",
    user_id: str = "",
) -> str:
    """Build the system prompt for the action agent with runtime context."""
    return ACTION_SYSTEM_PROMPT.format(
        current_time=datetime.now(timezone.utc).isoformat(),
        workspace_id=workspace_id or "not specified",
        project_id=project_id or "not specified",
        user_id=user_id or "unknown",
    )


@register_agent
class ActionAgent(BaseAgent):
    """Agent that performs workspace operations via RPM-BE API tools.

    This agent handles "action" requests — creating tasks, managing
    projects, searching content, generating summaries, etc.
    """

    name = "action"
    description = (
        "Performs workspace operations: create/manage tasks, search content, "
        "generate project summaries, assign work to team members, manage cycles "
        "and sticky notes. Use for any request that requires DOING something "
        "in the workspace rather than just answering a knowledge question."
    )

    def get_system_prompt(self, **kwargs) -> str:
        return build_action_prompt(**kwargs)

    async def run(self, state) -> dict:
        # The action agent is handled specially by the pipeline/chat.py
        # via create_react_agent, not through this method.
        # This exists for interface compliance.
        return {
            "agent_output": {
                "agent_name": self.name,
                "content": "Action agent runs via streaming pipeline.",
                "sources": [],
                "metadata": {},
            }
        }
