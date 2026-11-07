"""Builders producing Frontend `work_item_overview` & `metric_summary` widgets cleanly."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from ai.agents.project.gateway.contracts import ProjectAnalytics, WorkItemSummary


def _to_widget_work_item(item: WorkItemSummary) -> dict[str, Any]:
    return {
        "id": item.id,
        "title": f"{item.identifier}: {item.title}" if item.identifier else item.title,
        "priority": item.priority or "none",
        "assignee": item.assignee_name or "",
        "dueDate": item.due_date,
        "isOverdue": item.is_overdue,
        "completed": item.completed,
        "project": {"name": item.project_name} if item.project_name else None,
    }


def build_work_item_overview_widget(
    items: Sequence[WorkItemSummary],
    *,
    title: str = "Work Items Overview",
    group_by: Literal["state", "priority", "flat"] = "state",
    max_items: int = 30,
) -> dict[str, Any]:
    """Build a JSON-ready `WorkItemOverviewWidget` payload matching Frontend `WidgetsPart.tsx`."""
    sliced = list(items[:max_items])
    done = sum(1 for w in sliced if w.completed)
    overdue = sum(1 for w in sliced if w.is_overdue and not w.completed)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for w in sliced:
        if group_by == "priority":
            label = (w.priority or "none").capitalize()
        elif group_by == "flat":
            label = "Work Items"
        else:
            label = w.state_name or (w.state_group.capitalize() if w.state_group else "Other")
        grouped.setdefault(label, []).append(_to_widget_work_item(w))

    return {
        "type": "work_item_overview",
        "title": title,
        "subtitle": f"{len(sliced)} item(s) displayed ({done} done, {overdue} overdue)",
        "total": len(sliced),
        "done": done,
        "inProgress": max(0, len(sliced) - done),
        "overdue": overdue,
        "groups": [{"label": lbl, "workItems": rows} for lbl, rows in grouped.items()],
    }


def build_metric_summary_widget(
    analytics: ProjectAnalytics,
    *,
    title: str = "Project Metrics",
    extra_metrics: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Build a JSON-ready `MetricSummaryWidget` payload matching Frontend `WidgetsPart.tsx`."""
    rate_pct = (
        round(analytics.completion_rate * 100)
        if analytics.completion_rate <= 1.0
        else round(analytics.completion_rate)
    )
    metrics: list[dict[str, Any]] = [
        {"label": "Total Items", "value": analytics.total_items, "tone": "default"},
        {"label": "Completed", "value": analytics.completed_items, "tone": "good"},
        {
            "label": "Overdue",
            "value": analytics.overdue_items,
            "tone": "warn" if analytics.overdue_items > 0 else "default",
        },
        {
            "label": "Completion",
            "value": f"{rate_pct}%",
            "tone": "good" if rate_pct >= 70 else "default",
        },
        *list(extra_metrics),
    ]
    return {
        "type": "metric_summary",
        "title": title,
        "metrics": metrics[:8],
    }
