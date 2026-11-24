"""Write commands for creating and updating work items, assignments, comments, and labels."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ai.agents.project.gateway.contracts import ProjectGatewayError, WorkItemSummary
from ai.agents.project.gateway.transport import (
    project_label_route,
    work_item_comment_route,
    work_item_route,
)
from ai.agents.project.tools.contracts import (
    ProjectToolResult,
    ProjectToolRuntime,
    ProjectToolSpec,
)
from ai.shared.resilience.mutations import (
    MutationRecord,
    is_turn_journal_bound,
    record_patch_undo,
    record_turn_mutation,
)


class CreateWorkItemArgs(BaseModel):
    title: str = Field(..., min_length=1, max_length=300, description="Work item title.")
    project_id: str | None = Field(
        default=None, description="Target project ID or name (defaults to active project)."
    )
    description: str | None = Field(default=None, description="Optional detailed description.")
    state: str | None = Field(
        default=None,
        description="Optional Kanban column name, group (backlog/started/...), or state ID.",
    )
    assignee: str | None = Field(
        default=None,
        description="Optional assignee name, email, or user ID.",
    )
    priority: Literal["urgent", "high", "medium", "low", "none"] | None = None
    due_date: str | None = Field(default=None, description="Optional ISO date/datetime.")
    start_date: str | None = Field(default=None, description="Optional ISO date/datetime.")
    labels: list[str] = Field(default_factory=list, description="Optional label names or IDs.")
    parent_work_item_id: str | None = Field(
        default=None, description="Optional parent work item UUID."
    )


class UpdateWorkItemArgs(BaseModel):
    work_item_id: str = Field(..., min_length=1, description="Work item UUID or identifier.")
    project_id: str | None = Field(
        default=None,
        description="Optional project ID/name used to resolve state/assignee names.",
    )
    title: str | None = None
    description: str | None = None
    state: str | None = Field(
        default=None,
        description="Move work item to another Kanban column name, group, or state ID.",
    )
    assignee: str | None = Field(
        default=None, description="Reassign by member name, email, or user ID."
    )
    clear_assignee: bool = Field(default=False, description="Set true to unassign the work item.")
    priority: Literal["urgent", "high", "medium", "low", "none"] | None = None
    due_date: str | None = None
    start_date: str | None = None
    completed: bool | None = None
    labels: list[str] | None = Field(
        default=None, description="Replace labels by list of label names or IDs."
    )


class AssignWorkItemArgs(BaseModel):
    work_item_id: str = Field(..., min_length=1, description="Work item UUID or identifier.")
    assignee: str | None = Field(
        default=None,
        description="Assignee name/email/user ID; omit or null to unassign.",
    )
    project_id: str | None = None


class AddCommentArgs(BaseModel):
    work_item_id: str = Field(..., min_length=1, description="Work item UUID or identifier.")
    content: str = Field(..., min_length=1, description="Markdown/text comment body.")
    project_id: str | None = None


class CreateLabelArgs(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Label name.")
    project_id: str | None = None
    color: str | None = Field(default=None, description="Optional hex color (e.g. '#3b82f6').")
    description: str | None = None


async def _resolve_work_item_project(
    runtime: ProjectToolRuntime,
    work_item_id: str,
    explicit_project: str | None,
) -> tuple[str | None, str | None]:
    if explicit_project or runtime.scope.project_id:
        return await runtime.resolve_project_id(explicit_project)
    item = await runtime.gateway.work_items.get_work_item(work_item_id)
    if item.project_id:
        return item.project_id, None
    return await runtime.resolve_project_id(None)


async def handle_create_work_item(
    runtime: ProjectToolRuntime, args: CreateWorkItemArgs
) -> ProjectToolResult:
    pid, err = await runtime.resolve_project_id(args.project_id)
    if err or not pid:
        return ProjectToolResult(tool="create_work_item", status="error", error=err)

    column_id: str | None = None
    if args.state:
        column_id, st_err = await runtime.resolve_state_id(pid, args.state)
        if st_err:
            return ProjectToolResult(tool="create_work_item", status="error", error=st_err)

    assignee_id: str | None = None
    if args.assignee:
        assignee_id, mem_err = await runtime.resolve_member_id(pid, args.assignee)
        if mem_err:
            return ProjectToolResult(tool="create_work_item", status="error", error=mem_err)

    label_ids: list[str] = []
    raw_label_names: list[str] = []
    if args.labels:
        label_ids, raw_label_names = await runtime.resolve_label_ids(pid, args.labels)

    created = await runtime.gateway.work_items.create_work_item(
        pid,
        title=args.title,
        description=args.description,
        column_id=column_id,
        assignee_id=assignee_id,
        priority=args.priority,
        due_date=args.due_date,
        start_date=args.start_date,
        label_ids=label_ids,
        labels=raw_label_names,
        parent_work_item_id=args.parent_work_item_id,
    )
    if created.id.strip():
        record_turn_mutation(
            MutationRecord(
                tool="create_work_item",
                entity_type="work_item",
                entity_id=created.id,
                label=created.title or args.title,
                undo_method="DELETE",
                undo_path=work_item_route(created.id),
            )
        )
    return ProjectToolResult(
        tool="create_work_item",
        status="ok",
        data={"work_item": created.model_dump(mode="json")},
    )


async def _safe_get_work_item(
    runtime: ProjectToolRuntime, work_item_id: str
) -> WorkItemSummary | None:
    try:
        return await runtime.gateway.work_items.get_work_item(work_item_id)
    except (ProjectGatewayError, TimeoutError, OSError):
        return None


def _work_item_before_state(before: WorkItemSummary, args: UpdateWorkItemArgs) -> dict[str, Any]:
    snap: dict[str, Any] = {}
    if args.title is not None:
        snap["title"] = before.title
    if args.description is not None:
        snap["description"] = before.description
    if args.state is not None and before.column_id is not None:
        snap["columnId"] = before.column_id
    if args.assignee is not None or args.clear_assignee:
        snap["assigneeId"] = before.assignee_id
    if args.priority is not None:
        snap["priority"] = before.priority
    if args.due_date is not None:
        snap["dueDate"] = before.due_date
    if args.start_date is not None:
        snap["startDate"] = before.start_date
    if args.completed is not None:
        snap["completed"] = before.completed
    if args.labels is not None:
        snap["labelIds"] = list(before.label_ids)
    return snap


async def handle_update_work_item(
    runtime: ProjectToolRuntime, args: UpdateWorkItemArgs
) -> ProjectToolResult:
    column_id: str | None = None
    assignee_id: str | None = None
    label_ids: list[str] | None = None
    label_names: list[str] | None = None

    before: WorkItemSummary | None = None
    if is_turn_journal_bound():
        before = await _safe_get_work_item(runtime, args.work_item_id)

    needs_project = bool(args.state or args.assignee or args.labels is not None)
    pid: str | None = None
    if needs_project:
        pid, err = await _resolve_work_item_project(runtime, args.work_item_id, args.project_id)
        if err or not pid:
            return ProjectToolResult(tool="update_work_item", status="error", error=err)

    if args.state and pid:
        column_id, st_err = await runtime.resolve_state_id(pid, args.state)
        if st_err:
            return ProjectToolResult(tool="update_work_item", status="error", error=st_err)

    if args.assignee and pid:
        assignee_id, mem_err = await runtime.resolve_member_id(pid, args.assignee)
        if mem_err:
            return ProjectToolResult(tool="update_work_item", status="error", error=mem_err)

    if args.labels is not None and pid:
        label_ids, label_names = await runtime.resolve_label_ids(pid, args.labels)

    updated = await runtime.gateway.work_items.update_work_item(
        args.work_item_id,
        title=args.title,
        description=args.description,
        column_id=column_id,
        assignee_id=assignee_id,
        clear_assignee=args.clear_assignee,
        priority=args.priority,
        due_date=args.due_date,
        start_date=args.start_date,
        completed=args.completed,
        label_ids=label_ids if label_ids else None,
        labels=label_names if label_names else None,
    )
    if before is not None and updated.id.strip():
        before_state = _work_item_before_state(before, args)
        if before_state:
            record_patch_undo(
                tool="update_work_item",
                entity_type="work_item",
                entity_id=updated.id,
                label=updated.title or args.title or args.work_item_id,
                path=work_item_route(updated.id),
                before_state=before_state,
                method="PUT",
            )
    return ProjectToolResult(
        tool="update_work_item",
        status="ok",
        data={"work_item": updated.model_dump(mode="json")},
    )


async def handle_assign_work_item(
    runtime: ProjectToolRuntime, args: AssignWorkItemArgs
) -> ProjectToolResult:
    pid, err = await _resolve_work_item_project(runtime, args.work_item_id, args.project_id)
    if err or not pid:
        return ProjectToolResult(tool="assign_work_item", status="error", error=err)

    assignee_id: str | None = None
    if args.assignee and args.assignee.strip():
        assignee_id, mem_err = await runtime.resolve_member_id(pid, args.assignee)
        if mem_err:
            return ProjectToolResult(tool="assign_work_item", status="error", error=mem_err)

    res = await runtime.gateway.work_items.assign_work_item(
        pid,
        args.work_item_id,
        assignee_id=assignee_id,
    )
    return ProjectToolResult(tool="assign_work_item", status="ok", data=res)


async def handle_add_comment(
    runtime: ProjectToolRuntime, args: AddCommentArgs
) -> ProjectToolResult:
    pid = runtime.scope.project_id
    if args.project_id:
        pid, err = await runtime.resolve_project_id(args.project_id)
        if err:
            return ProjectToolResult(tool="add_work_item_comment", status="error", error=err)

    comment = await runtime.gateway.work_items.add_comment(
        args.work_item_id,
        content=args.content,
        project_id=pid,
    )
    if comment.id.strip():
        record_turn_mutation(
            MutationRecord(
                tool="add_work_item_comment",
                entity_type="comment",
                entity_id=comment.id,
                label=args.content[:40],
                undo_method="DELETE",
                undo_path=work_item_comment_route(comment.id),
            )
        )
    return ProjectToolResult(
        tool="add_work_item_comment",
        status="ok",
        data={"comment": comment.model_dump(mode="json")},
    )


async def handle_create_label(
    runtime: ProjectToolRuntime, args: CreateLabelArgs
) -> ProjectToolResult:
    pid, err = await runtime.resolve_project_id(args.project_id)
    if err or not pid:
        return ProjectToolResult(tool="create_label", status="error", error=err)

    label = await runtime.gateway.projects.create_label(
        pid,
        name=args.name,
        color=args.color,
        description=args.description,
    )
    if label.id.strip():
        record_turn_mutation(
            MutationRecord(
                tool="create_label",
                entity_type="label",
                entity_id=label.id,
                label=label.name or args.name,
                undo_method="DELETE",
                undo_path=project_label_route(pid, label.id),
            )
        )
    cached = runtime.labels_by_project.get(pid)
    if cached is not None:
        cached.append(label)
    return ProjectToolResult(
        tool="create_label",
        status="ok",
        data={"label": label.model_dump(mode="json")},
    )


def _confirm_create_work_item(args: dict[str, Any]) -> str:
    return f"Create work item `{args.get('title') or 'Untitled'}`"


def _confirm_update_work_item(args: dict[str, Any]) -> str:
    return f"Update work item `{args.get('work_item_id') or ''}`"


def _confirm_assign_work_item(args: dict[str, Any]) -> str:
    target = args.get("assignee") or "unassigned"
    return f"Assign work item `{args.get('work_item_id') or ''}` to `{target}`"


def _confirm_add_comment(args: dict[str, Any]) -> str:
    return f"Post comment on work item `{args.get('work_item_id') or ''}`"


def _confirm_create_label(args: dict[str, Any]) -> str:
    return f"Create label `{args.get('name') or ''}`"


def build_write_tools() -> tuple[ProjectToolSpec, ...]:
    """Build reversible write commands."""
    return (
        ProjectToolSpec(
            name="create_work_item",
            description=(
                "Create a new work item in a project (resolves state column, assignee name, "
                "and labels automatically)."
            ),
            args_schema=CreateWorkItemArgs,
            risk_tier="write",
            idempotent=False,
            handler=handle_create_work_item,
            describe_confirmation=_confirm_create_work_item,
        ),
        ProjectToolSpec(
            name="update_work_item",
            description=(
                "Update work item fields or move its Kanban column (`state`), priority, due date, "
                "labels, or completion status."
            ),
            args_schema=UpdateWorkItemArgs,
            risk_tier="write",
            idempotent=False,
            handler=handle_update_work_item,
            describe_confirmation=_confirm_update_work_item,
        ),
        ProjectToolSpec(
            name="assign_work_item",
            description="Assign or unassign a work item using member name, email, or user ID.",
            args_schema=AssignWorkItemArgs,
            risk_tier="write",
            idempotent=False,
            handler=handle_assign_work_item,
            describe_confirmation=_confirm_assign_work_item,
        ),
        ProjectToolSpec(
            name="add_work_item_comment",
            description="Post a discussion comment on a work item.",
            args_schema=AddCommentArgs,
            risk_tier="write",
            idempotent=False,
            handler=handle_add_comment,
            describe_confirmation=_confirm_add_comment,
        ),
        ProjectToolSpec(
            name="create_label",
            description="Create a new work-item label in a project.",
            args_schema=CreateLabelArgs,
            risk_tier="write",
            idempotent=False,
            handler=handle_create_label,
            describe_confirmation=_confirm_create_label,
        ),
    )
