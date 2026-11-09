"""Deterministic LaTeX context normalizer, viewport windower, and prompt formatter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ai.agents.editor.context.contracts import CompileErrorItem, EditorContextSnapshot
from ai.agents.editor.context.preamble import extract_local_symbols, extract_preamble_summary


def _to_int_or_none(val: Any) -> int | None:
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str) and val.strip().lstrip("-").isdigit():
        return int(val.strip())
    return None


def _clean_text(val: Any) -> str:
    if val is None:
        return ""
    return str(val).replace("\r\n", "\n").replace("\r", "\n")


def normalize_compile_errors(raw: Any, *, limit: int = 15) -> tuple[CompileErrorItem, ...]:
    """Normalize compiler diagnostic payloads (list, dict `{errors, warnings}`, or string)."""
    items: list[CompileErrorItem] = []

    def _push(entry: Any, default_severity: str = "error") -> None:
        if len(items) >= limit or entry is None:
            return
        if isinstance(entry, str):
            text = entry.strip()
            if text:
                items.append(CompileErrorItem(severity=default_severity, message=text[:400]))
            return
        if isinstance(entry, dict):
            msg = str(entry.get("message") or entry.get("error") or entry.get("description") or "").strip()
            if not msg:
                return
            items.append(
                CompileErrorItem(
                    line=_to_int_or_none(entry.get("line") or entry.get("startLine")),
                    file=str(entry.get("file") or entry.get("filePath") or "").strip().lstrip("/"),
                    severity=str(entry.get("severity") or default_severity).strip().lower() or default_severity,
                    message=msg[:400],
                    context=str(entry.get("context") or entry.get("rawExcerpt") or "").strip()[:300],
                )
            )

    if isinstance(raw, str):
        for line in raw.splitlines():
            if line.strip():
                _push(line.strip())
    elif isinstance(raw, list | tuple):
        for entry in raw:
            _push(entry)
    elif isinstance(raw, dict):
        errs = raw.get("errors")
        warns = raw.get("warnings")
        if isinstance(errs, list | tuple):
            for entry in errs:
                _push(entry, "error")
        if isinstance(warns, list | tuple):
            for entry in warns:
                _push(entry, "warning")
        if not errs and not warns and ("message" in raw or "error" in raw):
            _push(raw)

    return tuple(items)


def _format_numbered_lines(lines: Sequence[str], *, start_line: int, max_chars: int) -> tuple[str, int]:
    if not lines:
        return "", start_line - 1
    width = max(3, len(str(start_line + len(lines) - 1)))
    rendered: list[str] = []
    used_chars = 0
    last_line = start_line - 1

    for offset, line in enumerate(lines):
        line_num = start_line + offset
        formatted = f"L{line_num:0{width}d}: {line}"
        step_cost = len(formatted) + (1 if rendered else 0)
        if rendered and used_chars + step_cost > max_chars:
            break
        rendered.append(formatted)
        used_chars += step_cost
        last_line = line_num

    return "\n".join(rendered), last_line


def build_editor_context_snapshot(
    raw_context: Mapping[str, Any] | None,
    *,
    active_content_override: str | None = None,
    viewport_radius_lines: int = 45,
    max_chars: int = 16_000,
) -> EditorContextSnapshot:
    """Build a bounded, line-numbered snapshot of the active LaTeX file and IDE focus."""
    ctx = raw_context or {}
    page_id = str(ctx.get("page_id") or "").strip()
    active_file_id = str(ctx.get("active_file_id") or "").strip() or None
    active_file_path = (
        str(ctx.get("active_file_path") or ctx.get("filename") or "").strip().lstrip("/")
        or "main.tex"
    )

    content = _clean_text(
        active_content_override if active_content_override is not None else ctx.get("file_content")
    )
    lines = content.split("\n") if content else []
    total_lines = len(lines)

    cursor_line = _to_int_or_none(ctx.get("cursor_line"))
    cursor_column = _to_int_or_none(ctx.get("cursor_column"))
    raw_user_sel = ctx.get("user_selection")
    raw_selection = ctx.get("selection") or (
        raw_user_sel if isinstance(raw_user_sel, str) else ""
    )
    selection_text = _clean_text(raw_selection).strip("\n")
    sel_start = _to_int_or_none(ctx.get("selection_start_line"))
    sel_end = _to_int_or_none(ctx.get("selection_end_line"))
    current_section = str(ctx.get("current_section") or "").strip()
    current_environment = str(ctx.get("current_environment") or "").strip()
    structure_summary = _clean_text(
        ctx.get("document_structure_summary") or ctx.get("document_structure")
    ).strip()[:3000]
    compile_errors = normalize_compile_errors(ctx.get("compile_errors"))

    preamble = extract_preamble_summary(content)
    local_labels, local_citations = extract_local_symbols(content)

    viewport_start = 1
    viewport_end = total_lines
    is_truncated = False
    viewport_numbered = ""

    if total_lines > 0:
        if total_lines <= 220 and len(content) <= max_chars:
            viewport_numbered, viewport_end = _format_numbered_lines(
                lines,
                start_line=1,
                max_chars=max_chars,
            )
            is_truncated = viewport_end < total_lines
        else:
            focus_start = sel_start or cursor_line or (compile_errors[0].line if compile_errors else None) or 1
            focus_end = sel_end if (sel_end is not None and sel_end >= focus_start) else focus_start
            viewport_start = max(1, focus_start - viewport_radius_lines)
            window_end = min(total_lines, focus_end + viewport_radius_lines)
            slice_lines = lines[viewport_start - 1 : window_end]
            viewport_numbered, viewport_end = _format_numbered_lines(
                slice_lines,
                start_line=viewport_start,
                max_chars=max_chars,
            )
            is_truncated = viewport_start > 1 or viewport_end < total_lines

    return EditorContextSnapshot(
        page_id=page_id,
        active_file_id=active_file_id,
        active_file_path=active_file_path,
        total_lines=total_lines,
        cursor_line=cursor_line,
        cursor_column=cursor_column,
        selection_text=selection_text[:6000],
        selection_start_line=sel_start,
        selection_end_line=sel_end,
        current_section=current_section,
        current_environment=current_environment,
        document_structure_summary=structure_summary,
        preamble=preamble,
        local_labels=local_labels[:60],
        local_citations=local_citations[:60],
        compile_errors=compile_errors,
        viewport_start_line=viewport_start if total_lines > 0 else 1,
        viewport_end_line=viewport_end,
        is_truncated=is_truncated,
        viewport_numbered=viewport_numbered,
    )


def format_context_for_prompt(snapshot: EditorContextSnapshot) -> str:
    """Render an `EditorContextSnapshot` into a compact, unambiguous prompt block."""
    meta_lines: list[str] = [
        f"- Active file: `{snapshot.active_file_path}`"
        + (f" (file_id={snapshot.active_file_id})" if snapshot.active_file_id else "")
        + (f" — {snapshot.total_lines} lines" if snapshot.total_lines else " — (content not loaded)"),
    ]
    if snapshot.page_id:
        meta_lines.append(f"- Active page_id: `{snapshot.page_id}`")
    if snapshot.cursor_line is not None:
        col_part = f", column {snapshot.cursor_column}" if snapshot.cursor_column is not None else ""
        meta_lines.append(f"- Cursor position: line {snapshot.cursor_line}{col_part}")
    if snapshot.current_section:
        meta_lines.append(f"- Current section: `{snapshot.current_section}`")
    if snapshot.current_environment:
        meta_lines.append(f"- Inside environment: `\\begin{{{snapshot.current_environment}}}`")

    if snapshot.preamble.document_class or snapshot.preamble.packages:
        cls_text = snapshot.preamble.document_class or "unknown"
        pkg_text = ", ".join(snapshot.preamble.packages[:25]) or "none detected"
        meta_lines.append(f"- Preamble: `\\documentclass{{{cls_text}}}` | packages: [{pkg_text}]")
    if snapshot.preamble.custom_macros:
        meta_lines.append(
            f"- Custom macros in active file: {', '.join(snapshot.preamble.custom_macros[:20])}"
        )
    if snapshot.local_labels:
        meta_lines.append(f"- Local `\\label`s: {', '.join(snapshot.local_labels[:25])}")
    if snapshot.local_citations:
        meta_lines.append(f"- Cited keys in active file: {', '.join(snapshot.local_citations[:25])}")

    sections: list[str] = ["## Active Editor Context\n" + "\n".join(meta_lines)]

    if snapshot.preamble.raw_excerpt and snapshot.is_truncated:
        sections.append(
            "### Preamble Excerpt\n```latex\n" + snapshot.preamble.raw_excerpt + "\n```"
        )

    if snapshot.selection_text:
        span = ""
        if snapshot.selection_start_line is not None:
            end = snapshot.selection_end_line or snapshot.selection_start_line
            span = f" (L{snapshot.selection_start_line}-L{end})"
        sections.append(
            f"### User Selected Text{span}\n```latex\n{snapshot.selection_text}\n```"
        )

    if snapshot.compile_errors:
        err_lines = []
        for err in snapshot.compile_errors[:10]:
            loc = f"{err.file or snapshot.active_file_path}"
            if err.line is not None:
                loc += f":L{err.line}"
            err_lines.append(f"- [{err.severity.upper()}] ({loc}) {err.message}")
        sections.append("### Recent Compile Diagnostics\n" + "\n".join(err_lines))

    if snapshot.document_structure_summary:
        sections.append(f"### Document Structure\n{snapshot.document_structure_summary}")

    if snapshot.viewport_numbered:
        window_note = (
            f"lines {snapshot.viewport_start_line}..{snapshot.viewport_end_line} of {snapshot.total_lines}"
            if snapshot.is_truncated
            else f"full file, {snapshot.total_lines} lines"
        )
        sections.append(
            f"### Active File Content (`{snapshot.active_file_path}`, {window_note})\n"
            f"```latex\n{snapshot.viewport_numbered}\n```"
        )

    return "\n\n".join(sections)
