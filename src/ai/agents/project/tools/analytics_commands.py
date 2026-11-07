"""Tier-0 analytics, workload, overdue, member summary & project health commands."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ai.agents.project.analytics.health_audit import audit_project_health
from ai.agents.project.analytics.widget_builder import (
    build_metric_summary_widget,
    build_work_item_overview_widget,
)
from ai.agents.project.gateway.contracts import ProjectAnalytics
from ai.agents.project.resolution.resolver import resolve_member
from ai.agents.project.tools.contracts import (
    ProjectToolResult,
    ProjectToolRuntime,
    ProjectToolSpec,
)
from ai.agents.project.tools.read_commands import ProjectRefArgs


class SummarizeMemberArgs(BaseModel):
    member: str = Field(
        ...,
        min_length=1,
        description="Member name, email, or user ID to summarize workload for.",
    )
    project_id: str | None = Field(
        default=None,
        description="Project ID or name (defaults to active project).",
    )


async def handle_get_workload_distribution(
    runtime: ProjectToolRuntime, args: ProjectRefArgs
) -> ProjectToolResult:
    pid, err = await runtime.resolve_project_id(args.project_id)
    if err or not pid:
        return ProjectToolResult(tool="get_workload_distribution", status="error", error=err)
    analytics = await runtime.gateway.analytics.get_project_analytics(runtime.scope, pid)
    runtime.add_widget(build_metric_summary_widget(analytics, title="Workload Summary"))
    return ProjectToolResult(
        tool="get_workload_distribution",
        status="ok",
        data=analytics.model_dump(mode="json"),
    )


async def handle_get_overdue_report(
    runtime: ProjectToolRuntime, args: ProjectRefArgs
) -> ProjectToolResult:
    pid, err = await runtime.resolve_project_id(args.project_id)
    if err or not pid:
        return ProjectToolResult(tool="get_overdue_report", status="error", error=err)
    items = await runtime.gateway.work_items.list_work_items(
        runtime.scope, pid, completed=False, limit=200
    )
    overdue = [w for w in items if w.is_overdue and not w.completed]
    runtime.add_widget(
        build_work_item_overview_widget(overdue, title="Overdue Work Items")
    )
    return ProjectToolResult(
        tool="get_overdue_report",
        status="ok",
        data={
            "project_id": pid,
            "total_overdue": len(overdue),
            "overdue_items": [w.model_dump(mode="json") for w in overdue[:40]],
        },
    )


async def handle_audit_project_health(
    runtime: ProjectToolRuntime, args: ProjectRefArgs
) -> ProjectToolResult:
    pid, err = await runtime.resolve_project_id(args.project_id)
    if err or not pid:
        return ProjectToolResult(tool="audit_project_health", status="error", error=err)

    items = await runtime.gateway.work_items.list_work_items(runtime.scope, pid, limit=200)
    analytics = await runtime.gateway.analytics.get_project_analytics(runtime.scope, pid)
    report = audit_project_health(items, analytics, project_id=pid)
    runtime.add_widget(
        build_metric_summary_widget(
            analytics,
            title="Project Health Summary",
            extra_metrics=(
                {
                    "label": "Health Score",
                    "value": f"{report.score}/100",
                    "tone": (
                        "good"
                        if report.score >= 80
                        else ("warn" if report.score >= 50 else "bad")
                    ),
                },
            ),
        )
    )
    return ProjectToolResult(
        tool="audit_project_health",
        status="ok",
        data=report.model_dump(mode="json"),
    )


async def handle_summarize_member_work_items(
    runtime: ProjectToolRuntime, args: SummarizeMemberArgs
) -> ProjectToolResult:
    pid, err = await runtime.resolve_project_id(args.project_id)
    if err or not pid:
        return ProjectToolResult(tool="summarize_member_work_items", status="error", error=err)

    members = await runtime.get_members(pid)
    mem_res = resolve_member(args.member, members)
    if not mem_res.ok or mem_res.match is None:
        return ProjectToolResult(
            tool="summarize_member_work_items",
            status="error",
            error=mem_res.error,
        )

    items = await runtime.gateway.work_items.list_work_items(
        runtime.scope, pid, assignee_id=mem_res.match.id, limit=200
    )
    completed = sum(1 for w in items if w.completed)
    overdue = sum(1 for w in items if w.is_overdue and not w.completed)
    in_progress = sum(1 for w in items if w.state_group == "started" and not w.completed)
    rate = completed / len(items) if items else 0.0

    summary_analytics = ProjectAnalytics(
        project_id=pid,
        total_items=len(items),
        completed_items=completed,
        in_progress_items=in_progress,
        overdue_items=overdue,
        completion_rate=rate,
    )
    runtime.add_widget(
        build_metric_summary_widget(
            summary_analytics,
            title=f"Workload Summary — {mem_res.match.label}",
        )
    )
    runtime.add_widget(
        build_work_item_overview_widget(items, title=f"Assigned to {mem_res.match.label}")
    )
    return ProjectToolResult(
        tool="summarize_member_work_items",
        status="ok",
        data={
            "member_id": mem_res.match.id,
            "member_name": mem_res.match.label,
            "total_work_items": len(items),
            "completed_work_items": completed,
            "in_progress_work_items": in_progress,
            "overdue_work_items": overdue,
            "completion_rate": round(rate, 3),
            "work_items": [w.model_dump(mode="json") for w in items[:30]],
        },
    )


def build_analytics_tools() -> tuple[ProjectToolSpec, ...]:
    """Build Tier-0 KPI analytics, overdue, member workload & health audit tools."""
    return (
        ProjectToolSpec(
            name="get_workload_distribution",
            description="Get KPI counts, state breakdown, priority distribution & assignee workload.",
            args_schema=ProjectRefArgs,
            risk_tier="read",
            idempotent=True,
            handler=handle_get_workload_distribution,
        ),
        ProjectToolSpec(
            name="get_overdue_report",
            description="List all overdue, incomplete work items in a project.",
            args_schema=ProjectRefArgs,
            risk_tier="read",
            idempotent=True,
            handler=handle_get_overdue_report,
        ),
        ProjectToolSpec(
            name="audit_project_health",
            description=(
                "Run structural Project Health Audit checking overdue items, unassigned high-priority "
                "risks, assignee workload imbalance, and WIP bottlenecks."
            ),
            args_schema=ProjectRefArgs,
            risk_tier="read",
            idempotent=True,
            handler=handle_audit_project_health,
        ),
        ProjectToolSpec(
            name="summarize_member_work_items",
            description="Summarize workload, completion rate, and assigned items for one team member.",
            args_schema=SummarizeMemberArgs,
            risk_tier="read",
            idempotent=True,
            handler=handle_summarize_member_work_items,
        ),
    )
