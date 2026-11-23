"""Prompt builders for `project_agent` planning, entity accuracy, and grounded synthesis."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any


def build_planner_prompt(tool_catalog: str, *, step: int, max_steps: int) -> str:
    """Build strict JSON planner prompt ordered Static-First, Volatile-Last for KV Prefix Caching."""
    return (
        "You are an expert Flux Research Project Management & Planning specialist.\n\n"
        "## Available Tools\n"
        f"{tool_catalog}\n\n"
        "## Output Contract\n"
        "Do NOT invoke native tool calls; return plain JSON text only (no `<think>` blocks, no markdown fences):\n"
        '{"action": "call_tools" | "respond", '
        '"tool_calls": [{"tool": "<tool_name>", "args": {...}, "reason": "<brief>"}], '
        '"answer": "<final markdown answer if responding directly with 0 tool calls else empty>", '
        '"reasoning": "<brief>"}\n\n'
        "## Planning & Grounding Rules\n"
        "1. Rely strictly on Active Project Context, verified tool outputs, and Findings from Peer "
        "Specialists (e.g. `rag_agent` or `editor_agent`). Never fabricate project IDs, work item "
        "identifiers, member names, workflow columns, or metrics. You ONLY manage Kanban work items, "
        "projects, members, labels, stickies, and analytics — NEVER convert a request to create or "
        "initialize a LaTeX Page/article (`tạo bài viết`, `khởi tạo bài viết`, `create page`) into "
        "`create_work_item`.\n"
        "2. Check Active Project Context first before calling discovery tools unnecessarily; use "
        "`list_projects` only when multiple projects exist or the active project is unspecified.\n"
        "3. You can pass human-readable names/identifiers directly to tools (`state='In Progress'`, "
        "`assignee='Minh'`, `labels=['Bug']`). If a tool reports ambiguity, ask the user cleanly "
        "or clarify with exact names.\n"
        "4. Use analytics tools (`get_workload_distribution`, `get_overdue_report`, "
        "`audit_project_health`, `summarize_member_work_items`) or `list_work_items` / "
        "`get_my_work_items` when summarizing progress, risks, overdue tasks, or member workload.\n"
        "5. Destructive actions (`delete_work_item`, `remove_project_member`, `update_member_role`, "
        "`update_project`, `delete_label`) require user confirmation via HITL; call one clean action "
        "with exact parameters and respect user rejection without retrying silently.\n"
        "6. Once all needed tool calls have completed (or user rejected a confirmation), emit "
        '"action": "respond" with "answer": "" so the streaming synthesizer presents the result.\n\n'
        f"Current step: {step}/{max_steps}."
    )


def build_synthesizer_prompt(*, has_widgets: bool) -> str:
    """Build concise, structured synthesizer prompt grounded in tool observations."""
    rules: list[str] = [
        (
            "1. State only facts, work item identifiers, statuses, assignees, dates, and numbers "
            "observed in the Active Project Context, Peer Findings, and tool observations below."
        ),
        "2. Reply in the exact language of the user's request.",
        (
            "3. When Peer Findings or tool calls failed or returned errors, report clearly what "
            "failed or was rejected without pretending actions succeeded."
        ),
    ]
    if has_widgets:
        rules.append(
            "4. Interactive Overview/Metric Widgets are automatically attached in the UI from "
            "verified tool outputs. Highlight the key takeaways, risks, and next actions concisely "
            "without dumping redundant long raw JSON tables."
        )
    return (
        "You are the Flux Project Management assistant.\n"
        "Synthesize a clear, concise, actionable response for the user.\n\n"
        "## Rules\n" + "\n".join(rules)
    )


def format_observations_block(
    *,
    query: str,
    context_snapshot_text: str,
    observations: Sequence[dict[str, Any]],
    peer_findings: Sequence[dict[str, Any]] = (),
) -> str:
    """Assemble query, hydrated project metadata, peer agent outputs, and tool observations."""
    sections: list[str] = [f"User request: {query}"]

    if context_snapshot_text.strip():
        sections.append(context_snapshot_text.strip())

    if peer_findings:
        peer_lines: list[str] = ["## Findings from Peer Specialists"]
        for finding in peer_findings:
            agent = str(finding.get("agent") or "peer")
            content = str(finding.get("content") or "").strip()
            peer_lines.append(f"### From `{agent}`\n{content}")
        sections.append("\n\n".join(peer_lines))

    if not observations:
        sections.append("Tool observations: (none yet)")
    else:
        rendered = json.dumps(list(observations), ensure_ascii=False, indent=2)
        if len(rendered) > 14_000:
            rendered = rendered[-14_000:]
        sections.append(f"Tool observations so far:\n{rendered}")

    return "\n\n".join(sections)
