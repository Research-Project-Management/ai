"""Tier-0 read-only Project, Work Item, Member, Label & Activity commands."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ai.agents.project.analytics.widget_builder import build_work_item_overview_widget
from ai.agents.project.resolution.resolver import (
    resolve_label,
    resolve_member,
    resolve_workflow_state,
)
from ai.agents.project.tools.contracts import (
    ProjectToolResult,
    ProjectToolRuntime,
    ProjectToolSpec,
)


class ProjectRefArgs(BaseModel):
    project_id: str | None = Field(
        default=None,
        description="Project ID, identifier, or name (defaults to active project).",
    )


class ListProjectsArgs(BaseModel):
    project_type: Literal["created", "shared", "all"] = "all"
    search: str = Field(default="", description="Optional name/identifier search filter.")
    priority: str | None = Field(default=None, description="Optional urgent/high/medium/low/none.")
    is_archived: bool | None = None


class ListWorkItemsArgs(BaseModel):
    project_id: str | None = Field(
        default=None,
        description="Target project ID or name (defaults to active project).",
    )
    state: str | None = Field(
        default=None,
        description="Optional Kanban column name, group (started/completed/...), or ID.",
    )
    assignee: str | None = Field(
        default=None,
        description="Optional assignee name, email, or user ID.",
    )
    label: str | None = Field(default=None, description="Optional label name or ID.")
    priority: Literal["urgent", "high", "medium", "low", "none"] | None = None
    completed: bool | None = None
    search: str = ""
    limit: int = Field(default=40, ge=1, le=100)


class GetMyWorkItemsArgs(BaseModel):
    project_id: str | None = Field(
        default=None,
        description="Optional project filter (defaults to all projects when omitted).",
    )
    relation: Literal["assigned", "created", "subscribed", "all"] = "assigned"


class GetWorkItemArgs(BaseModel):
    work_item_id: str = Field(
        ...,
        min_length=1,
        description="Work item UUID or identifier (e.g. 'FLUX-42').",
    )
    include_comments: bool = True


class ListMembersArgs(BaseModel):
    project_id: str | None = None
    role: Literal["owner", "coordinator", "contributor", "reviewer"] | None = None
    search: str = ""


class ActivityFeedArgs(BaseModel):
    project_id: str | None = Field(
        default=None,
        description="Optional project ID/name; uses workspace/user feed when omitted.",
    )
    entity_type: str | None = Field(
        default=None,
        description="Optional filter: work_item, page, project, sticky, comment.",
    )
    limit: int = Field(default=25, ge=1, le=60)


async def handle_list_projects(
    runtime: ProjectToolRuntime, args: ListProjectsArgs
) -> ProjectToolResult:
    projects = await runtime.gateway.projects.list_projects(
        runtime.scope,
        project_type=args.project_type,
        search=args.search,
        priority=args.priority,
        is_archived=args.is_archived,
    )
    if (
        not args.search
        and not args.priority
        and args.is_archived is None
        and args.project_type == "all"
    ):
        runtime.projects = list(projects)
    return ProjectToolResult(
        tool="list_projects",
        status="ok",
        data={
            "total": len(projects),
            "projects": [p.model_dump(mode="json") for p in projects[:30]],
        },
    )


async def handle_get_project_overview(
    runtime: ProjectToolRuntime, args: ProjectRefArgs
) -> ProjectToolResult:
    pid, err = await runtime.resolve_project_id(args.project_id)
    if err or not pid:
        return ProjectToolResult(tool="get_project_overview", status="error", error=err)
    overview = await runtime.gateway.projects.get_project_overview(runtime.scope, pid)
    return ProjectToolResult(
        tool="get_project_overview",
        status="ok",
        data=overview.model_dump(mode="json"),
    )


async def handle_list_work_items(
    runtime: ProjectToolRuntime, args: ListWorkItemsArgs
) -> ProjectToolResult:
    pid, err = await runtime.resolve_project_id(args.project_id)
    if err or not pid:
        return ProjectToolResult(tool="list_work_items", status="error", error=err)

    column_id: str | None = None
    if args.state:
        states = await runtime.get_states(pid)
        st_res = resolve_workflow_state(args.state, states)
        if not st_res.ok or st_res.match is None:
            return ProjectToolResult(tool="list_work_items", status="error", error=st_res.error)
        column_id = st_res.match.id

    assignee_id: str | None = None
    if args.assignee:
        members = await runtime.get_members(pid)
        mem_res = resolve_member(args.assignee, members)
        if not mem_res.ok or mem_res.match is None:
            return ProjectToolResult(tool="list_work_items", status="error", error=mem_res.error)
        assignee_id = mem_res.match.id

    label_id: str | None = None
    if args.label:
        labels = await runtime.get_labels(pid)
        lb_res = resolve_label(args.label, labels)
        if not lb_res.ok or lb_res.match is None:
            return ProjectToolResult(tool="list_work_items", status="error", error=lb_res.error)
        label_id = lb_res.match.id

    items = await runtime.gateway.work_items.list_work_items(
        runtime.scope,
        pid,
        column_id=column_id,
        priority=args.priority,
        assignee_id=assignee_id,
        label=label_id,
        completed=args.completed,
        search=args.search,
        limit=args.limit,
    )
    runtime.add_widget(build_work_item_overview_widget(items, title="Project Work Items Overview"))
    return ProjectToolResult(
        tool="list_work_items",
        status="ok",
        data={
            "project_id": pid,
            "total": len(items),
            "work_items": [w.model_dump(mode="json") for w in items],
        },
    )


async def handle_get_my_work_items(
    runtime: ProjectToolRuntime, args: GetMyWorkItemsArgs
) -> ProjectToolResult:
    pid: str | None = None
    if args.project_id:
        pid, err = await runtime.resolve_project_id(args.project_id)
        if err:
            return ProjectToolResult(tool="get_my_work_items", status="error", error=err)

    items = await runtime.gateway.work_items.get_my_work_items(
        runtime.scope,
        project_id=pid,
        relation=args.relation,
    )
    runtime.add_widget(build_work_item_overview_widget(items, title="My Work Items Overview"))
    return ProjectToolResult(
        tool="get_my_work_items",
        status="ok",
        data={
            "total": len(items),
            "relation": args.relation,
            "work_items": [w.model_dump(mode="json") for w in items],
        },
    )


async def handle_get_work_item(
    runtime: ProjectToolRuntime, args: GetWorkItemArgs
) -> ProjectToolResult:
    item = await runtime.gateway.work_items.get_work_item(runtime.scope, args.work_item_id)
    comments = (
        await runtime.gateway.work_items.list_comments(runtime.scope, item.id or args.work_item_id)
        if args.include_comments
        else []
    )
    return ProjectToolResult(
        tool="get_work_item",
        status="ok",
        data={
            "work_item": item.model_dump(mode="json"),
            "comments": [c.model_dump(mode="json") for c in comments[:25]],
        },
    )


async def handle_list_members(
    runtime: ProjectToolRuntime, args: ListMembersArgs
) -> ProjectToolResult:
    pid, err = await runtime.resolve_project_id(args.project_id)
    if err or not pid:
        return ProjectToolResult(tool="list_members", status="error", error=err)
    members = await runtime.gateway.projects.list_members(
        runtime.scope,
        pid,
        role=args.role,
        search=args.search,
        limit=100,
    )
    if not args.role and not args.search:
        runtime.members_by_project[pid] = list(members)
    return ProjectToolResult(
        tool="list_members",
        status="ok",
        data={
            "project_id": pid,
            "total": len(members),
            "members": [m.model_dump(mode="json") for m in members],
        },
    )


async def handle_list_labels(
    runtime: ProjectToolRuntime, args: ProjectRefArgs
) -> ProjectToolResult:
    pid, err = await runtime.resolve_project_id(args.project_id)
    if err or not pid:
        return ProjectToolResult(tool="list_labels", status="error", error=err)
    labels = await runtime.get_labels(pid)
    return ProjectToolResult(
        tool="list_labels",
        status="ok",
        data={
            "project_id": pid,
            "total": len(labels),
            "labels": [lb.model_dump(mode="json") for lb in labels],
        },
    )


async def handle_get_workspace_activity(
    runtime: ProjectToolRuntime, args: ActivityFeedArgs
) -> ProjectToolResult:
    pid: str | None = runtime.scope.project_id
    if args.project_id:
        pid, err = await runtime.resolve_project_id(args.project_id)
        if err:
            return ProjectToolResult(tool="get_workspace_activity", status="error", error=err)

    entries = await runtime.gateway.projects.list_activity(
        runtime.scope,
        project_id=pid,
        entity_type=args.entity_type,
        limit=args.limit,
    )
    return ProjectToolResult(
        tool="get_workspace_activity",
        status="ok",
        data={
            "project_id": pid,
            "total": len(entries),
            "items": [e.model_dump(mode="json") for e in entries],
        },
    )


def build_read_tools() -> tuple[ProjectToolSpec, ...]:
    """Build Tier-0 read-only Project inspection commands."""
    return (
        ProjectToolSpec(
            name="list_projects",
            description="List all projects accessible to the user with IDs, roles, and priorities.",
            args_schema=ListProjectsArgs,
            risk_tier="read",
            idempotent=True,
            handler=handle_list_projects,
        ),
        ProjectToolSpec(
            name="get_project_overview",
            description=(
                "Fetch high-level project dashboard metrics, lead, status updates, and recent activity."
            ),
            args_schema=ProjectRefArgs,
            risk_tier="read",
            idempotent=True,
            handler=handle_get_project_overview,
        ),
        ProjectToolSpec(
            name="list_work_items",
            description=(
                "List/filter work items in a project by column/state name, assignee name, label, "
                "priority, completion, or search query; attaches an interactive overview widget."
            ),
            args_schema=ListWorkItemsArgs,
            risk_tier="read",
            idempotent=True,
            handler=handle_list_work_items,
        ),
        ProjectToolSpec(
            name="get_my_work_items",
            description="List work items assigned to, created by, or subscribed by the current user.",
            args_schema=GetMyWorkItemsArgs,
            risk_tier="read",
            idempotent=True,
            handler=handle_get_my_work_items,
        ),
        ProjectToolSpec(
            name="get_work_item",
            description="Inspect full details, state, assignee, labels, and comments of one work item.",
            args_schema=GetWorkItemArgs,
            risk_tier="read",
            idempotent=True,
            handler=handle_get_work_item,
        ),
        ProjectToolSpec(
            name="list_members",
            description="List project team members with roles, user IDs, and emails.",
            args_schema=ListMembersArgs,
            risk_tier="read",
            idempotent=True,
            handler=handle_list_members,
        ),
        ProjectToolSpec(
            name="list_labels",
            description="List work-item labels defined in a project.",
            args_schema=ProjectRefArgs,
            risk_tier="read",
            idempotent=True,
            handler=handle_list_labels,
        ),
        ProjectToolSpec(
            name="get_workspace_activity",
            description="Fetch recent activity feed events for a project or across the workspace.",
            args_schema=ActivityFeedArgs,
            risk_tier="read",
            idempotent=True,
            handler=handle_get_workspace_activity,
        ),
    )
