"""Compact formatter turning hydrated project metadata into Planner prompt context."""

from __future__ import annotations

from ai.agents.project.resolution.contracts import ProjectContextSnapshot


def format_project_context(snapshot: ProjectContextSnapshot) -> str:
    """Render active project, columns, team roster, labels, and accessible projects cleanly."""
    lines: list[str] = ["## Active Project Management Context"]

    if snapshot.user_id:
        lines.append(f"- Current User ID: {snapshot.user_id}")
    if snapshot.space_id:
        lines.append(f"- Space ID: {snapshot.space_id}")

    if snapshot.active_project is not None:
        p = snapshot.active_project
        ident = f" [{p.identifier}]" if p.identifier else ""
        lines.append(
            f"- Active Project: {p.name}{ident} (id={p.id}, role={p.your_role or 'member'}, "
            f"priority={p.priority})"
        )
    elif snapshot.projects:
        proj_preview = ", ".join(
            f"{p.name}[{p.identifier or p.id[:8]}]" for p in snapshot.projects[:12]
        )
        lines.append(f"- Accessible Projects ({len(snapshot.projects)}): {proj_preview}")
    else:
        lines.append("- Active Project: (none scoped yet — call `list_projects` first)")

    if snapshot.workflow_states:
        state_items = [
            f"{s.name} [group={s.group}, id={s.id}]" for s in snapshot.workflow_states[:20]
        ]
        lines.append("- Workflow Columns: " + "; ".join(state_items))

    if snapshot.members:
        member_items = [
            f"{m.name} ({m.role}, id={m.user_id})" for m in snapshot.members[:25]
        ]
        lines.append("- Project Members: " + "; ".join(member_items))

    if snapshot.labels:
        label_items = [f"{lb.name} (id={lb.id})" for lb in snapshot.labels[:30]]
        lines.append("- Project Labels: " + ", ".join(label_items))

    return "\n".join(lines)
