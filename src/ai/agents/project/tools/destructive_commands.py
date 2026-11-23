"""Tier-2 HITL-gated destructive & admin commands (`risk_tier='destructive'`)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ai.agents.project.resolution.resolver import resolve_label, resolve_member
from ai.agents.project.tools.contracts import (
    ProjectToolResult,
    ProjectToolRuntime,
    ProjectToolSpec,
)


class DeleteWorkItemArgs(BaseModel):
    work_item_id: str = Field(..., min_length=1, description="Work item UUID or identifier.")
    mode: Literal["archive", "delete"] = Field(
        default="archive",
        description="Choose 'archive' (soft-archive) or 'delete' (move to trash).",
    )


class UpdateProjectArgs(BaseModel):
    project_id: str | None = Field(
        default=None, description="Target project ID or name (defaults to active project)."
    )
    name: str | None = None
    description: str | None = None
    priority: Literal["urgent", "high", "medium", "low", "none"] | None = None
    start_date: str | None = None
    target_date: str | None = None
    is_archived: bool | None = None


class AddMemberArgs(BaseModel):
    user_id: str = Field(..., min_length=1, description="User UUID to add to the project.")
    role: Literal["coordinator", "contributor", "reviewer"] = "contributor"
    project_id: str | None = None


class UpdateMemberRoleArgs(BaseModel):
    member: str = Field(..., min_length=1, description="Member name, email, or user ID.")
    role: Literal["coordinator", "contributor", "reviewer"]
    project_id: str | None = None


class RemoveMemberArgs(BaseModel):
    member: str = Field(..., min_length=1, description="Member name, email, or user ID to remove.")
    project_id: str | None = None


class DeleteLabelArgs(BaseModel):
    label: str = Field(..., min_length=1, description="Label name or ID to delete.")
    project_id: str | None = None


async def handle_delete_work_item(
    runtime: ProjectToolRuntime, args: DeleteWorkItemArgs
) -> ProjectToolResult:
    if args.mode == "archive":
        item = await runtime.gateway.work_items.archive_work_item(runtime.scope, args.work_item_id)
        return ProjectToolResult(
            tool="delete_work_item",
            status="ok",
            data={"mode": "archive", "work_item": item.model_dump(mode="json")},
        )
    res = await runtime.gateway.work_items.delete_work_item(runtime.scope, args.work_item_id)
    return ProjectToolResult(
        tool="delete_work_item",
        status="ok",
        data={"mode": "delete", "result": res},
    )


async def handle_update_project(
    runtime: ProjectToolRuntime, args: UpdateProjectArgs
) -> ProjectToolResult:
    pid, err = await runtime.resolve_project_id(args.project_id)
    if err or not pid:
        return ProjectToolResult(tool="update_project", status="error", error=err)
    updated = await runtime.gateway.projects.update_project(
        runtime.scope,
        pid,
        name=args.name,
        description=args.description,
        priority=args.priority,
        start_date=args.start_date,
        target_date=args.target_date,
        is_archived=args.is_archived,
    )
    return ProjectToolResult(
        tool="update_project",
        status="ok",
        data={"project": updated.model_dump(mode="json")},
    )


async def handle_add_member(runtime: ProjectToolRuntime, args: AddMemberArgs) -> ProjectToolResult:
    pid, err = await runtime.resolve_project_id(args.project_id)
    if err or not pid:
        return ProjectToolResult(tool="add_project_member", status="error", error=err)
    member = await runtime.gateway.projects.add_member(
        runtime.scope, pid, user_id=args.user_id, role=args.role
    )
    runtime.members_by_project.pop(pid, None)
    return ProjectToolResult(
        tool="add_project_member",
        status="ok",
        data={"member": member.model_dump(mode="json")},
    )


async def handle_update_member_role(
    runtime: ProjectToolRuntime, args: UpdateMemberRoleArgs
) -> ProjectToolResult:
    pid, err = await runtime.resolve_project_id(args.project_id)
    if err or not pid:
        return ProjectToolResult(tool="update_member_role", status="error", error=err)

    mem_res = resolve_member(args.member, await runtime.get_members(pid))
    if not mem_res.ok or mem_res.match is None:
        return ProjectToolResult(tool="update_member_role", status="error", error=mem_res.error)

    updated = await runtime.gateway.projects.update_member_role(
        runtime.scope, pid, mem_res.match.id, role=args.role
    )
    runtime.members_by_project.pop(pid, None)
    return ProjectToolResult(
        tool="update_member_role",
        status="ok",
        data={"member": updated.model_dump(mode="json")},
    )


async def handle_remove_member(
    runtime: ProjectToolRuntime, args: RemoveMemberArgs
) -> ProjectToolResult:
    pid, err = await runtime.resolve_project_id(args.project_id)
    if err or not pid:
        return ProjectToolResult(tool="remove_project_member", status="error", error=err)

    mem_res = resolve_member(args.member, await runtime.get_members(pid))
    if not mem_res.ok or mem_res.match is None:
        return ProjectToolResult(tool="remove_project_member", status="error", error=mem_res.error)

    res = await runtime.gateway.projects.remove_member(runtime.scope, pid, mem_res.match.id)
    runtime.members_by_project.pop(pid, None)
    return ProjectToolResult(
        tool="remove_project_member",
        status="ok",
        data={"removed_user_id": mem_res.match.id, "removed_name": mem_res.match.label, **res},
    )


async def handle_delete_label(
    runtime: ProjectToolRuntime, args: DeleteLabelArgs
) -> ProjectToolResult:
    pid, err = await runtime.resolve_project_id(args.project_id)
    if err or not pid:
        return ProjectToolResult(tool="delete_label", status="error", error=err)

    lb_res = resolve_label(args.label, await runtime.get_labels(pid))
    if not lb_res.ok or lb_res.match is None:
        return ProjectToolResult(tool="delete_label", status="error", error=lb_res.error)

    res = await runtime.gateway.projects.delete_label(runtime.scope, pid, lb_res.match.id)
    runtime.labels_by_project.pop(pid, None)
    return ProjectToolResult(
        tool="delete_label",
        status="ok",
        data={"deleted_label_id": lb_res.match.id, "deleted_label": lb_res.match.label, **res},
    )


def _confirm_delete_work_item(args: dict[str, Any]) -> str:
    mode = str(args.get("mode") or "archive")
    verb = "Archive" if mode == "archive" else "Delete"
    return f"{verb} work item `{args.get('work_item_id') or ''}`"


def _confirm_update_project(args: dict[str, Any]) -> str:
    target = args.get("project_id") or "active project"
    return f"Update project settings for `{target}`"


def _confirm_add_member(args: dict[str, Any]) -> str:
    return f"Add user `{args.get('user_id') or ''}` as `{args.get('role') or 'contributor'}`"


def _confirm_update_role(args: dict[str, Any]) -> str:
    return f"Change role of `{args.get('member') or ''}` to `{args.get('role') or ''}`"


def _confirm_remove_member(args: dict[str, Any]) -> str:
    return f"Remove member `{args.get('member') or ''}` from project and unassign their items"


def _confirm_delete_label(args: dict[str, Any]) -> str:
    return f"Delete label `{args.get('label') or ''}` from project"


def build_destructive_tools() -> tuple[ProjectToolSpec, ...]:
    """Build Tier-2 HITL-gated destructive and project administration tools."""
    return (
        ProjectToolSpec(
            name="delete_work_item",
            description="HITL-gated destructive action: archive or delete a work item.",
            args_schema=DeleteWorkItemArgs,
            risk_tier="destructive",
            idempotent=False,
            handler=handle_delete_work_item,
            describe_confirmation=_confirm_delete_work_item,
        ),
        ProjectToolSpec(
            name="update_project",
            description=(
                "HITL-gated admin action: update project name, description, priority, dates, or archive."
            ),
            args_schema=UpdateProjectArgs,
            risk_tier="destructive",
            idempotent=False,
            handler=handle_update_project,
            describe_confirmation=_confirm_update_project,
        ),
        ProjectToolSpec(
            name="add_project_member",
            description="HITL-gated admin action: add a user to the project with a specified role.",
            args_schema=AddMemberArgs,
            risk_tier="destructive",
            idempotent=False,
            handler=handle_add_member,
            describe_confirmation=_confirm_add_member,
        ),
        ProjectToolSpec(
            name="update_member_role",
            description="HITL-gated admin action: change a project member's role.",
            args_schema=UpdateMemberRoleArgs,
            risk_tier="destructive",
            idempotent=False,
            handler=handle_update_member_role,
            describe_confirmation=_confirm_update_role,
        ),
        ProjectToolSpec(
            name="remove_project_member",
            description=(
                "HITL-gated destructive action: remove a member from the project (unassigns their items)."
            ),
            args_schema=RemoveMemberArgs,
            risk_tier="destructive",
            idempotent=False,
            handler=handle_remove_member,
            describe_confirmation=_confirm_remove_member,
        ),
        ProjectToolSpec(
            name="delete_label",
            description="HITL-gated destructive action: delete a label and detach it from work items.",
            args_schema=DeleteLabelArgs,
            risk_tier="destructive",
            idempotent=False,
            handler=handle_delete_label,
            describe_confirmation=_confirm_delete_label,
        ),
    )
