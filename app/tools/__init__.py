"""
Tools package — LangChain @tool wrappers for RPM-BE API endpoints.

Each tool corresponds to a workspace operation the agent can perform
on behalf of the user. Tools are auto-imported and aggregated into
ALL_TOOLS for the ReAct agent.
"""

from app.tools.task_tools import (
    list_tasks,
    create_task,
    update_task,
    delete_task,
)
from app.tools.project_tools import (
    list_projects,
    get_project_overview,
    get_project_details,
)
from app.tools.workspace_tools import (
    search_workspace,
    list_members,
    get_workspace_overview,
    get_workspace_activity,
)
from app.tools.cycle_tools import (
    list_cycles,
    create_cycle,
)
from app.tools.page_tools import (
    list_pages,
    get_page_content,
)
from app.tools.sticky_tools import (
    list_stickies,
    create_sticky,
)

# Master list — passed to the ReAct agent
ALL_TOOLS = [
    # Task management
    list_tasks,
    create_task,
    update_task,
    delete_task,
    # Project
    list_projects,
    get_project_overview,
    get_project_details,
    # Workspace
    search_workspace,
    list_members,
    get_workspace_overview,
    get_workspace_activity,
    # Research cycles
    list_cycles,
    create_cycle,
    # Pages (read-only for Phase 1)
    list_pages,
    get_page_content,
    # Sticky notes
    list_stickies,
    create_sticky,
]

# Categorize for permission checking
SAFE_TOOLS = {t.name for t in ALL_TOOLS if not any(
    kw in t.name for kw in ("delete", "remove", "update")
)}
DESTRUCTIVE_TOOLS = {t.name for t in ALL_TOOLS} - SAFE_TOOLS
