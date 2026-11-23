"""Prompt builders for `library_agent` action planning and grounded synthesis."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any


def build_planner_prompt(
    tool_catalog: str,
    *,
    step: int,
    max_steps: int,
) -> str:
    """Build strict JSON planner prompt ordered Static-First, Volatile-Last for KV Prefix Caching."""
    return (
        "You are a precision Flux Library specialist agent.\n\n"
        "## Available Tools\n"
        f"{tool_catalog}\n\n"
        "## Output Contract\n"
        "Do NOT invoke any native function or tool calls; return plain JSON text only.\n"
        "Output ONLY a valid JSON object immediately (no `<think>` blocks, no markdown fences, no extra text):\n"
        '{"action": "call_tools" | "respond", '
        '"tool_calls": [{"tool": "<tool_name>", "args": {...}, "reason": "<brief>"}], '
        '"answer": "<grounded markdown answer when action=respond, else empty>", '
        '"reasoning": "<brief>"}\n\n'
        "## Rules\n"
        "1. Always inspect or search with read tools (`search_items`, `library_stats`, `get_item`, "
        "`audit_library`) BEFORE calling citation, organization, trash, or merge tools unless the exact "
        "target UUID was already given in Attached library items, conversation, or prior observations. "
        "When target `item_id` values appear in Attached library items, use them directly.\n"
        "2. Before calling destructive tools (`trash_items`, `merge_duplicates`), resolve real titles and IDs "
        "first in an earlier step so the user sees accurate titles during confirmation.\n"
        "3. Never invent UUIDs, counts, titles, DOIs, or BibTeX entries.\n"
        "4. Preserve the user's language in your reasoning notes and final answer.\n"
        "5. Once you have sufficient tool observations (or cannot proceed further), return "
        '"action": "respond" with "answer": "" so the final response synthesizer streams '
        "the complete answer to the user.\n\n"
        f"Current step: {step}/{max_steps}."
    )


def build_synthesizer_prompt() -> str:
    """Build grounded response synthesis prompt from real Library tool observations."""
    return (
        "You are the Flux Library assistant.\n"
        "Answer the user strictly using the tool observations below.\n"
        "Rules:\n"
        "1. Never fabricate item IDs, counts, publication years, DOIs, BibTeX entries, or action outcomes.\n"
        "2. Reply in the same language as the user's request.\n"
        "3. Format lists of papers, counts, or duplicate clusters cleanly using Markdown tables when helpful.\n"
        "4. Preserve formatted citations or BibTeX/RIS exports verbatim in clean lists or fenced code blocks.\n"
        "5. If any tool returned status='rejected' or status='error', explicitly tell the user what was "
        "cancelled or failed instead of claiming it succeeded."
    )


def format_observations_block(
    *,
    query: str,
    observations: Sequence[dict[str, Any]],
    attached_items: Sequence[dict[str, Any]] = (),
) -> str:
    """Serialize attached items and compact tool observations alongside the user query."""
    sections: list[str] = [f"Current request: {query}"]
    if attached_items:
        rendered_items = json.dumps(list(attached_items), ensure_ascii=False, indent=2)
        sections.append(f"Attached library items:\n{rendered_items}")
    if not observations:
        sections.append("Tool observations: (none yet)")
    else:
        rendered = json.dumps(list(observations), ensure_ascii=False, indent=2)
        if len(rendered) > 14_000:
            rendered = rendered[-14_000:]
        sections.append(f"Tool observations so far:\n{rendered}")
    return "\n\n".join(sections)
