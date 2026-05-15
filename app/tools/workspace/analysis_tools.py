"""Analysis tools — compute insights from workspace data (read-only, no API side-effects)."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from collections import defaultdict

from langchain_core.tools import tool

from app.tools._client import flux_api, get_workspace_id


@tool
async def get_workload_distribution(project_id: str) -> dict:
    """Analyze task distribution across team members to identify overloaded members.

    Use this when the user asks:
    - "Ai đang bận nhất trong project?"
    - "Phân bổ công việc có đều không?"
    - "Workload distribution of the team"

    Returns task count per assignee, overdue tasks per person, and unassigned tasks.

    Args:
        project_id: MongoDB ObjectId of the project.
    """
    result = await flux_api.get(f"/api/project/{project_id}/tasks")
    if isinstance(result, dict) and result.get("error"):
        return result

    tasks = result.get("tasks", []) if isinstance(result, dict) else []
    distribution: dict[str, dict] = {}
    unassigned = 0
    unassigned_overdue = 0

    for t in tasks:
        assignee = t.get("assignee")
        if not assignee:
            unassigned += 1
            if t.get("isOverdue") and t.get("columnId") != "done":
                unassigned_overdue += 1
            continue

        name = assignee.get("name", "Unknown")
        uid = str(assignee.get("_id", name))

        if uid not in distribution:
            distribution[uid] = {
                "name": name,
                "total": 0,
                "done": 0,
                "overdue": 0,
                "in_progress": 0,
            }

        dist = distribution[uid]
        dist["total"] += 1
        col = t.get("columnId", "")
        if col == "done":
            dist["done"] += 1
        elif "progress" in col.lower() or "doing" in col.lower():
            dist["in_progress"] += 1

        if t.get("isOverdue") and col != "done":
            dist["overdue"] += 1

    members = sorted(distribution.values(), key=lambda x: x["total"], reverse=True)
    avg = sum(m["total"] for m in members) / len(members) if members else 0

    return {
        "project": result.get("projectName", project_id),
        "members": members,
        "unassigned_tasks": unassigned,
        "unassigned_overdue": unassigned_overdue,
        "average_tasks_per_member": round(avg, 1),
        "most_loaded": members[0]["name"] if members else None,
    }


@tool
async def get_overdue_report(project_id: str) -> dict:
    """Get a report of all overdue tasks in a project.

    Use this when the user asks:
    - "Task nào đang quá hạn?"
    - "Báo cáo task trễ deadline"
    - "Which tasks are overdue?"

    Returns tasks grouped by assignee with due date info.

    Args:
        project_id: MongoDB ObjectId of the project.
    """
    result = await flux_api.get(f"/api/project/{project_id}/tasks")
    if isinstance(result, dict) and result.get("error"):
        return result

    tasks = result.get("tasks", []) if isinstance(result, dict) else []
    overdue_tasks = [
        t for t in tasks
        if t.get("isOverdue") and t.get("columnId", "") not in ("done", "completed")
    ]

    grouped: dict[str, list] = defaultdict(list)
    for t in overdue_tasks:
        assignee = t.get("assignee")
        name = assignee.get("name", "Unassigned") if assignee else "Unassigned"
        grouped[name].append({
            "title": t.get("title"),
            "id": str(t.get("_id", "")),
            "dueDate": t.get("dueDate"),
            "priority": t.get("priority", "none"),
            "columnId": t.get("columnId"),
        })

    return {
        "project": result.get("projectName", project_id),
        "total_overdue": len(overdue_tasks),
        "by_assignee": dict(grouped),
    }


@tool
async def get_project_velocity(project_id: str) -> dict:
    """Analyze how many tasks were completed per week in a project.

    Use this when the user asks about team productivity or velocity trends.

    Args:
        project_id: MongoDB ObjectId of the project.
    """
    result = await flux_api.get(f"/api/project/{project_id}/tasks")
    if isinstance(result, dict) and result.get("error"):
        return result

    tasks = result.get("tasks", []) if isinstance(result, dict) else []
    columns = result.get("columns", [])
    done_col_ids = {
        col.get("id") for col in columns
        if (col.get("title") or "").lower() in ("done", "completed")
    }

    now = datetime.now(timezone.utc)
    weeks: dict[str, int] = {}

    for t in tasks:
        if t.get("columnId") not in done_col_ids:
            continue
        updated = t.get("updatedAt")
        if not updated:
            continue
        try:
            dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            delta = (now - dt).days
            week_num = delta // 7
            if week_num > 8:  # only last 8 weeks
                continue
            label = f"Week -{week_num}" if week_num > 0 else "This week"
            weeks[label] = weeks.get(label, 0) + 1
        except Exception:
            continue

    total_done = sum(weeks.values())
    total = len(tasks)

    return {
        "project": result.get("projectName", project_id),
        "total_tasks": total,
        "total_done": total_done,
        "completion_rate": f"{round(total_done / total * 100)}%" if total else "0%",
        "velocity_by_week": weeks,
    }


@tool
async def generate_team_summary(workspace_id: str = "") -> dict:
    """Generate a comprehensive team summary across all projects in the workspace.

    Use this when the user asks:
    - "Tóm tắt công việc của team"
    - "Tiến độ tổng thể workspace"
    - "Team update / weekly digest"

    Fetches projects, counts tasks per project, and summarizes overall progress.

    Args:
        workspace_id: Workspace ID. If empty, uses current session workspace.
    """
    ws = workspace_id or get_workspace_id()
    if not ws:
        return {"error": True, "message": "workspace_id is required"}

    projects_result = await flux_api.get(f"/api/workspace/{ws}/projects")
    if isinstance(projects_result, dict) and projects_result.get("error"):
        return projects_result

    projects = projects_result.get("projects", []) if isinstance(projects_result, dict) else []

    summary = []
    for p in projects[:10]:  # limit to avoid timeout
        pid = str(p.get("_id", ""))
        tasks_result = await flux_api.get(f"/api/project/{pid}/tasks")
        if isinstance(tasks_result, dict) and not tasks_result.get("error"):
            tasks = tasks_result.get("tasks", [])
            columns = tasks_result.get("columns", [])
            done_ids = {
                col.get("id") for col in columns
                if (col.get("title") or "").lower() in ("done", "completed")
            }
            total = len(tasks)
            done = sum(1 for t in tasks if t.get("columnId") in done_ids)
            overdue = sum(
                1 for t in tasks
                if t.get("isOverdue") and t.get("columnId") not in done_ids
            )
            summary.append({
                "project": p.get("name"),
                "id": pid,
                "total_tasks": total,
                "done": done,
                "overdue": overdue,
                "progress": f"{round(done / total * 100)}%" if total else "0%",
            })

    return {
        "workspace": ws,
        "projects_summarized": len(summary),
        "projects": summary,
        "total_tasks": sum(p["total_tasks"] for p in summary),
        "total_done": sum(p["done"] for p in summary),
        "total_overdue": sum(p["overdue"] for p in summary),
    }
