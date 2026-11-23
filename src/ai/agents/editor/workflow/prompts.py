"""Prompt builders for `editor_agent` planning, LaTeX rigor, and grounded synthesis."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ai.orchestrator.citations import CITATION_INSTRUCTIONS


def build_planner_prompt(
    tool_catalog: str,
    *,
    step: int,
    max_steps: int,
) -> str:
    """Build strict JSON planner prompt ordered Static-First, Volatile-Last for KV Prefix Caching."""
    return (
        "You are an expert LaTeX page editor and scientific writing specialist.\n\n"
        "## Available Tools\n"
        f"{tool_catalog}\n\n"
        "## Output Contract\n"
        "Do NOT invoke any native function or tool calls; return plain JSON text only.\n"
        "Output ONLY a valid JSON object immediately (no `<think>` blocks, no markdown fences, no extra text):\n"
        '{"action": "call_tools" | "respond", '
        '"tool_calls": [{"tool": "<tool_name>", "args": {...}, "reason": "<brief>"}], '
        '"answer": "<final markdown answer if responding directly with 0 edits, else empty>", '
        '"reasoning": "<brief>"}\n\n'
        "## LaTeX Rigor & Grounding Rules\n"
        "1. When the user asks to create or initialize a new page/article/paper (e.g., 'khởi tạo bài viết', "
        "'tạo page', 'create page'), call `create_page` with a clear `title` (and optional `description`/starter "
        "`content`) — `create_page` works even when no `page_id` is currently open.\n"
        "2. Whenever the user asks to edit, rewrite, polish, fix, format, or insert LaTeX in an existing "
        "page, propose verified diffs via `propose_edit` instead of only dumping raw text—unless "
        "no file content is accessible or only an explanation was requested.\n"
        "2. Before calling `propose_edit` on a non-active file (or when `Active File Content` is truncated "
        "outside the needed region), call `list_files` / `read_file` first so `search_text` quotes "
        "verbatim lines.\n"
        "3. In `propose_edit`, NEVER include `L###: ` line-number prefixes in `search_text` or "
        "`replace_text`. Quote 2-5 exact lines cleanly so the anchor matches uniquely.\n"
        "4. Use `kind='insert_preamble'` when adding missing `\\usepackage{...}` declarations; NEVER "
        "place `\\usepackage` before `\\documentclass{...}` (all packages must follow `\\documentclass{...}` "
        "and precede `\\begin{document}`). On modern TeX Live, do NOT recommend deprecated "
        "`\\usepackage[utf8]{inputenc}`; for Vietnamese text under `pdfLaTeX` (e.g., ấ, ơ, ư, đ), use "
        "`\\usepackage[T5]{fontenc}` (NOT `T1`, which lacks Vietnamese glyphs), or recommend XeLaTeX / "
        "LuaLaTeX with `\\usepackage{fontspec}` for non-Latin scripts.\n"
        "5. Zero Hallucination: NEVER invent `\\cite{key}` keys, `\\ref{label}` names, DOIs, BibTeX "
        "entries, experimental results, or custom macros. Verify project keys/labels via "
        "`inspect_citations` or `get_outline_and_symbols` (or rely strictly on Peer Agent Findings / "
        "Active Editor Context). If no verified citation key exists in `.bib`, omit fabricated keys "
        "or call `resolve_citation` / `add_bibtex_entries` only with verified DOIs/BibTeX.\n"
        "6. Citation Reviewer, Structure Auditor & Collaboration: Use `scan_citation_gaps` when asked "
        "to audit/review citations or find unsupported claims; use `audit_page_structure` for "
        "project-wide outline balance, cross-reference (`\\ref`/`\\label`) integrity, `\\input` link "
        "checks, or missing `\\usepackage` audits; use `list_review_comments` before revising around "
        "collaborator feedback. Prefer `propose_edit` for interactive CodeMirror diffs unless the user "
        "explicitly asks for a `.bib` update (`add_bibtex_entries`, `resolve_citation`, "
        "`sync_library_citations`) or a Track Changes suggestion (`create_review_suggestion`).\n"
        "7. If `propose_edit` returns an anchor or LaTeX syntax error, fix your arguments using the error "
        "message and retry.\n"
        "8. Once ready to answer the user (patches verified, write done, or question answerable), return "
        '"action": "respond" with "answer": "" so the streaming synthesizer presents the response.\n\n'
        f"Current step: {step}/{max_steps}."
    )


def build_synthesizer_prompt(*, has_patches: bool, has_citations: bool) -> str:
    """Build grounded response synthesis prompt aligned with Flux Editor UX."""
    rules: list[str] = [
        (
            "1. Rely strictly on the Active Editor Context, verified tool observations, verified patches, "
            "and Peer Agent Findings below. Never fabricate citation keys, labels, compile outputs, or file paths."
        ),
        "2. Reply in the exact language of the user's request.",
        (
            "3. Use clean Markdown with LaTeX math (`$...$` / `$$...$$`) and concise fenced `latex` blocks "
            "only when illustrating formulas or answering explanation questions."
        ),
        "4. If a tool failed or could not verify a patch, state clearly what failed and what remains unverified.",
        (
            "5. Never suggest placing `\\usepackage` before `\\documentclass{...}`, and do not recommend "
            "deprecated `\\usepackage[utf8]{inputenc}`; for Vietnamese in pdfLaTeX recommend "
            "`\\usepackage[T5]{fontenc}` (not `T1`), or XeLaTeX/LuaLaTeX + `\\usepackage{fontspec}` "
            "for non-Latin scripts."
        ),
    ]
    if has_patches:
        rules.append(
            "6. Verified Diff Patches are attached cleanly via the interactive Patch Card for CodeMirror review "
            "(the user can inspect diff hunks and accept/reject safely). Summarize each proposed patch "
            "concisely (file path, line span, what changed and why, plus any warnings) — do NOT repeat "
            "the entire unmodified file."
        )
    if has_citations:
        rules.append(f"7. {CITATION_INSTRUCTIONS}")

    return (
        "You are the Flux LaTeX Editor assistant.\n"
        "Synthesize a clear, concise, technically rigorous response for the user.\n\n"
        "## Rules\n" + "\n".join(rules)
    )


def format_observations_block(
    *,
    query: str,
    context_snapshot_text: str,
    observations: Sequence[dict[str, Any]],
    patches: Sequence[dict[str, Any]] = (),
    peer_findings: Sequence[dict[str, Any]] = (),
) -> str:
    """Assemble request, active LaTeX context, peer findings, verified patches, and tool outputs."""
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

    if patches:
        patch_lines = ["## Verified Patch Proposals Ready for UI"]
        for idx, p in enumerate(patches, start=1):
            span = f"L{p.get('from_line', '?')}-L{p.get('to_line', '?')}"
            warn_note = f" | warnings: {p.get('warnings')}" if p.get("warnings") else ""
            patch_lines.append(
                f"{idx}. [{p.get('file_path', 'main.tex')} {span}] {p.get('title', 'Edit')} "
                f"(kind={p.get('kind', 'replace')}, strategy={p.get('match_strategy', 'exact')}{warn_note})"
            )
            rationale = str(p.get("rationale") or "").strip()
            if rationale:
                patch_lines.append(f"   Rationale: {rationale}")
        sections.append("\n".join(patch_lines))

    if not observations:
        sections.append("Tool observations: (none yet)")
    else:
        rendered = json.dumps(list(observations), ensure_ascii=False, indent=2)
        if len(rendered) > 14_000:
            rendered = rendered[-14_000:]
        sections.append(f"Tool observations so far:\n{rendered}")

    return "\n\n".join(sections)
