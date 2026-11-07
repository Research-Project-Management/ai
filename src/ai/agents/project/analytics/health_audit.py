"""Composable Rule Pipeline auditing project delivery health (0 LLM, 0 I/O)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from ai.agents.project.gateway.contracts import ProjectAnalytics, WorkItemSummary


class HealthFinding(BaseModel):
    """Single actionable project health risk discovered by a rule."""

    model_config = ConfigDict(frozen=True)

    severity: Literal["high", "medium", "low"]
    category: str
    summary: str
    affected_ids: tuple[str, ...] = ()


class ProjectHealthReport(BaseModel):
    """Structured health report combining KPIs and rule findings."""

    model_config = ConfigDict(frozen=True)

    project_id: str
    score: int
    status: Literal["healthy", "at_risk", "critical"]
    total_items: int
    overdue_count: int
    unassigned_high_priority_count: int
    findings: tuple[HealthFinding, ...] = ()


@dataclass(frozen=True)
class HealthAuditContext:
    """Immutable inputs evaluated by each `ProjectHealthRule`."""

    project_id: str
    items: tuple[WorkItemSummary, ...]
    analytics: ProjectAnalytics | None = None


class ProjectHealthRule(Protocol):
    """OCP rule contract returning 0 or more `HealthFinding` objects."""

    def evaluate(self, ctx: HealthAuditContext) -> tuple[HealthFinding, ...]: ...


class OverdueWorkItemsRule:
    """Flag open work items whose due date has already passed."""

    def evaluate(self, ctx: HealthAuditContext) -> tuple[HealthFinding, ...]:
        overdue = [w for w in ctx.items if w.is_overdue and not w.completed]
        if not overdue:
            return ()
        sev: Literal["high", "medium", "low"] = "high" if len(overdue) >= 3 else "medium"
        titles = ", ".join(w.identifier or w.title[:35] for w in overdue[:5])
        return (
            HealthFinding(
                severity=sev,
                category="overdue",
                summary=f"{len(overdue)} work item(s) overdue ({titles}).",
                affected_ids=tuple(w.id for w in overdue[:10]),
            ),
        )


class UnassignedPriorityRule:
    """Flag urgent or high-priority work items without any assignee."""

    def evaluate(self, ctx: HealthAuditContext) -> tuple[HealthFinding, ...]:
        urgent_unassigned = [
            w
            for w in ctx.items
            if not w.completed
            and w.priority in {"urgent", "high"}
            and not w.assignee_id
            and not w.assignee_ids
        ]
        if not urgent_unassigned:
            return ()
        titles = ", ".join(w.identifier or w.title[:35] for w in urgent_unassigned[:5])
        return (
            HealthFinding(
                severity="high",
                category="unassigned_priority",
                summary=f"{len(urgent_unassigned)} urgent/high item(s) have no assignee ({titles}).",
                affected_ids=tuple(w.id for w in urgent_unassigned[:10]),
            ),
        )


class WorkloadImbalanceRule:
    """Flag when one team member holds disproportionate active workload."""

    def evaluate(self, ctx: HealthAuditContext) -> tuple[HealthFinding, ...]:
        active_counts: dict[str, int] = {}
        for item in ctx.items:
            if item.completed or item.is_archived:
                continue
            if item.assignee_name:
                active_counts[item.assignee_name] = active_counts.get(item.assignee_name, 0) + 1
        if len(active_counts) < 2:
            return ()
        top_name, top_count = max(active_counts.items(), key=lambda kv: kv[1])
        total_assigned = sum(active_counts.values())
        share = top_count / max(1, total_assigned)
        if top_count >= 5 and share >= 0.6:
            return (
                HealthFinding(
                    severity="medium",
                    category="workload_imbalance",
                    summary=(
                        f"Workload concentrated on {top_name} "
                        f"({top_count}/{total_assigned} active items, {round(share * 100)}%)."
                    ),
                ),
            )
        return ()


class WipBottleneckRule:
    """Flag excessive concurrent work-in-progress items inflating cycle time."""

    def __init__(self, *, max_wip_ratio: float = 0.5, min_wip_items: int = 8) -> None:
        self._max_wip_ratio = max_wip_ratio
        self._min_wip_items = min_wip_items

    def evaluate(self, ctx: HealthAuditContext) -> tuple[HealthFinding, ...]:
        started = [w for w in ctx.items if w.state_group == "started" and not w.completed]
        open_items = [w for w in ctx.items if not w.completed and not w.is_archived]
        if len(started) >= self._min_wip_items and (
            len(started) / max(1, len(open_items)) >= self._max_wip_ratio
        ):
            return (
                HealthFinding(
                    severity="medium",
                    category="wip_bottleneck",
                    summary=(
                        f"{len(started)} items actively started out of {len(open_items)} open "
                        "(high WIP concentration)."
                    ),
                    affected_ids=tuple(w.id for w in started[:10]),
                ),
            )
        return ()


DEFAULT_HEALTH_RULES: tuple[ProjectHealthRule, ...] = (
    OverdueWorkItemsRule(),
    UnassignedPriorityRule(),
    WorkloadImbalanceRule(),
    WipBottleneckRule(),
)


def audit_project_health(
    items: Sequence[WorkItemSummary],
    analytics: ProjectAnalytics | None = None,
    *,
    project_id: str = "",
    rules: Sequence[ProjectHealthRule] = DEFAULT_HEALTH_RULES,
) -> ProjectHealthReport:
    """Evaluate all `ProjectHealthRule` checks and synthesize a normalized health score."""
    ctx = HealthAuditContext(
        project_id=project_id,
        items=tuple(items),
        analytics=analytics,
    )
    findings: list[HealthFinding] = []
    for rule in rules:
        findings.extend(rule.evaluate(ctx))

    penalty = sum(
        25 if f.severity == "high" else (12 if f.severity == "medium" else 5)
        for f in findings
    )
    score = max(0, 100 - penalty)
    status: Literal["healthy", "at_risk", "critical"] = (
        "healthy" if score >= 80 else ("at_risk" if score >= 50 else "critical")
    )

    overdue_count = sum(1 for w in items if w.is_overdue and not w.completed)
    unassigned_high = sum(
        1
        for w in items
        if not w.completed
        and w.priority in {"urgent", "high"}
        and not w.assignee_id
        and not w.assignee_ids
    )
    total_count = analytics.total_items if analytics and analytics.total_items > 0 else len(items)

    return ProjectHealthReport(
        project_id=project_id,
        score=score,
        status=status,
        total_items=total_count,
        overdue_count=overdue_count,
        unassigned_high_priority_count=unassigned_high,
        findings=tuple(findings),
    )
