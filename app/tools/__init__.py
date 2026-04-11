"""
Tools package — LangChain @tool wrappers for RPM-BE API endpoints.

Each tool corresponds to a workspace operation the agent can perform
on behalf of the user. Tools are auto-imported and aggregated into
ALL_TOOLS for the ReAct agent.
"""

from app.tools.task_tools import (
    list_tasks,
    get_my_tasks,
    summarize_member_tasks,
    create_task,
    update_task,
    delete_task,
)
from app.tools.project_tools import (
    list_projects,
    get_project_overview,
    get_project_details,
    update_project,
    add_project_member,
    remove_project_member,
    update_member_role,
)
from app.tools.workspace_tools import (
    search_workspace,
    list_members,
    get_workspace_overview,
    get_workspace_activity,
)
from app.tools.cycle_tools import (
    list_cycles,
    get_cycle_details,
    create_cycle,
    update_cycle,
    delete_cycle,
)
from app.tools.page_tools import (
    list_pages,
    get_page_content,
    create_page,
    update_page,
    delete_page,
    get_page_versions,
    list_workspace_pages,
)
from app.tools.tag_tools import (
    list_tags,
    create_tag,
    delete_tag,
)
from app.tools.sticky_tools import (
    list_stickies,
    create_sticky,
    update_sticky,
    delete_sticky,
)
from app.tools.comment_tools import (
    get_task_comments,
    add_task_comment,
    get_page_comments,
    add_page_comment,
)
from app.tools.user_tools import (
    get_current_user,
    search_users,
)
from app.tools.analysis_tools import (
    get_workload_distribution,
    get_overdue_report,
    get_project_velocity,
    generate_team_summary,
)

# Master list — passed to the ReAct agent
ALL_TOOLS = [
    # ─── Task management ──────────────────────────────────────────────────────
    list_tasks,
    get_my_tasks,
    summarize_member_tasks,
    create_task,
    update_task,
    delete_task,

    # ─── Project ──────────────────────────────────────────────────────────────
    list_projects,
    get_project_overview,
    get_project_details,
    update_project,
    add_project_member,
    remove_project_member,
    update_member_role,

    # ─── Workspace ────────────────────────────────────────────────────────────
    search_workspace,
    list_members,
    get_workspace_overview,
    get_workspace_activity,

    # ─── Research cycles ──────────────────────────────────────────────────────
    list_cycles,
    get_cycle_details,
    create_cycle,
    update_cycle,
    delete_cycle,

    # ─── Pages (LaTeX documents) ──────────────────────────────────────────────
    list_pages,
    get_page_content,
    create_page,
    update_page,
    delete_page,
    get_page_versions,
    list_workspace_pages,

    # ─── Tags ─────────────────────────────────────────────────────────────────
    list_tags,
    create_tag,
    delete_tag,

    # ─── Sticky notes ─────────────────────────────────────────────────────────
    list_stickies,
    create_sticky,
    update_sticky,
    delete_sticky,

    # ─── Comments ─────────────────────────────────────────────────────────────
    get_task_comments,
    add_task_comment,
    get_page_comments,
    add_page_comment,

    # ─── Users ────────────────────────────────────────────────────────────────
    get_current_user,
    search_users,

    # ─── Analysis (computed, read-only) ───────────────────────────────────────
    get_workload_distribution,
    get_overdue_report,
    get_project_velocity,
    generate_team_summary,
]

# Categorize for permission checking
SAFE_TOOLS = {t.name for t in ALL_TOOLS if not any(
    kw in t.name for kw in ("delete", "remove", "update")
)}
DESTRUCTIVE_TOOLS = {t.name for t in ALL_TOOLS} - SAFE_TOOLS
