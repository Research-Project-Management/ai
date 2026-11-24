"""Request-context prompt section renderers for the Supervisor prompt."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ai.shared.contracts.sources import AttachedSource, parse_attached_sources

ContextSectionRenderer = Callable[[Mapping[str, Any]], list[str]]


def format_attached_source_line(src: AttachedSource) -> str:
    """Format one attached Library or uploaded source entry for the Supervisor prompt."""
    label = src.title or src.item_id or src.effective_source_id or "Untitled"
    if src.kind in ("library", "paper"):
        if src.indexed:
            status = "full-text indexed"
        elif src.has_pdf or src.file_id:
            status = "PDF auto-indexable"
        else:
            status = "metadata/notes only"
        item_ref = src.item_id or "none"
        sid_ref = src.effective_source_id or "none"
        return f'  - [library] "{label}" (item_id={item_ref}, status={status}, source_id={sid_ref})'
    sid_ref = src.effective_source_id or "none"
    return f'  - [upload] "{label}" (source_id={sid_ref})'


def _as_seq(value: Any) -> Sequence[Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes) and value:
        return value
    return None


def render_attached_sources(request_context: Mapping[str, Any]) -> list[str]:
    """Render attached Library papers, uploaded PDFs, and extra workspace document IDs."""
    lines: list[str] = []
    attached = parse_attached_sources(request_context.get("attached_sources"))
    if attached:
        lines.append("- Attached context sources:")
        lines.extend(format_attached_source_line(src) for src in attached)
        lines.append(
            "  Routing guidance: These sources are available resources, not mandatory "
            "triggers. Only delegate when answering the user's request actually depends on "
            "their content or metadata: use `rag_agent` for full-text content on uploads or "
            "library papers marked `full-text indexed` / `PDF auto-indexable`, and "
            "`library_agent` (passing exact `item_id`) for `metadata/notes only` papers "
            "(`get_item`), citations, notes, tags, or collections. For downstream tasks "
            "based on an attached source (`project_agent` / `editor_agent`), read the source "
            'first with "needs_followup": true, then act on the next round.'
        )

    covered_ids = {
        value
        for src in attached
        for value in (src.effective_source_id, src.item_id, src.file_id)
        if value
    }
    doc_ids = _as_seq(request_context.get("document_ids"))
    if doc_ids is not None:
        extra_docs = [
            str(d).strip() for d in doc_ids if str(d).strip() and str(d).strip() not in covered_ids
        ]
        if extra_docs:
            joined = ", ".join(extra_docs)
            lines.append(
                f"- Attached space document_ids: [{joined}] "
                "(Available uploaded space documents — delegate to `rag_agent` only when "
                "the request depends on reading or inspecting their content.)"
            )
    return lines


def render_project_context(request_context: Mapping[str, Any]) -> list[str]:
    """Render active project ID and Kanban/Page delegation guidance."""
    project_id = str(request_context.get("project_id") or "").strip()
    if not project_id:
        return []
    return [
        (
            f"- Active project_id: {project_id} "
            "(Delegate Kanban work-item/project/member/workload/activity requests to `project_agent`. "
            "However, requests to create, initialize, or write a new Page/article/paper must still go to "
            "`editor_agent`. When paired with attached papers or LaTeX tasks, coordinate with "
            '`rag_agent` / `editor_agent` via `"needs_followup": true`.)'
        )
    ]


def render_editor_context(request_context: Mapping[str, Any]) -> list[str]:
    """Render active LaTeX editor file, cursor, selection, and compile-error cues."""
    lines: list[str] = []
    page_id = str(request_context.get("page_id") or "").strip()
    filename = str(
        request_context.get("active_file_path") or request_context.get("filename") or ""
    ).strip()
    has_file_content = bool(str(request_context.get("file_content") or "").strip())
    has_selection = bool(str(request_context.get("selection") or "").strip())
    has_compile_errors = bool(request_context.get("compile_errors"))

    if page_id:
        lines.append(f"- Active space page_id: {page_id}")
    if filename or has_file_content or has_selection or has_compile_errors or page_id:
        editor_cues: list[str] = []
        if filename:
            editor_cues.append(f"file={filename}")
        if request_context.get("cursor_line") is not None:
            editor_cues.append(f"cursor_line={request_context['cursor_line']}")
        if has_selection:
            editor_cues.append("has_selection=true")
        if has_compile_errors:
            editor_cues.append("has_compile_errors=true")
        cue_summary = ", ".join(editor_cues) if editor_cues else "active page"
        lines.append(
            f"- Active LaTeX editor context ({cue_summary}): "
            "Delegate LaTeX editing, academic polishing, error debugging, equation/section drafting, "
            "page explanation, citation-gap audits, multi-file structure/outline/cross-reference/"
            "package audits, collaborator review-comment handling, and `.tex`/`.bib` inspection to "
            "`editor_agent`. If the task also requires extracting evidence "
            "from attached PDFs (`rag_agent`) or finding papers/DOIs in Library/web (`library_agent` / "
            "`web_search_agent`) before citing or writing, dispatch those prerequisite agents first with "
            '`"needs_followup": true`, then delegate to `editor_agent`. Similarly, when the user asks to '
            "scan the page for unsupported claims and fetch missing references, run `editor_agent` "
            'with `"needs_followup": true` to identify gaps before dispatching `library_agent` / '
            "`web_search_agent`."
        )
    return lines


def render_external_scope(request_context: Mapping[str, Any]) -> list[str]:
    """Render domain filters, target URLs, client sources, and search mode cues."""
    allowed = _as_seq(
        request_context.get("allowed_domains") or request_context.get("include_domains")
    )
    blocked = _as_seq(
        request_context.get("blocked_domains") or request_context.get("exclude_domains")
    )
    target_urls = _as_seq(request_context.get("target_urls") or request_context.get("urls"))
    ext_sources = _as_seq(
        request_context.get("external_sources") or request_context.get("client_sources")
    )
    search_mode = str(request_context.get("search_mode") or "").strip()

    ext_signals: list[str] = []
    if allowed is not None:
        ext_signals.append(f"allowed_domains={list(allowed)}")
    if blocked is not None:
        ext_signals.append(f"blocked_domains={list(blocked)}")
    if target_urls is not None:
        ext_signals.append(f"target_urls={list(target_urls)}")
    if ext_sources is not None:
        ext_signals.append(f"external_sources={len(ext_sources)}")
    if search_mode:
        ext_signals.append(f"search_mode={search_mode}")

    if not ext_signals:
        return []
    return [
        (
            f"- External search/fetch scope ({'; '.join(ext_signals)}): "
            "External web/scholar scope or URLs are attached — delegate to `web_search_agent` "
            "rather than answering from memory."
        )
    ]


def render_space_memory(request_context: Mapping[str, Any]) -> list[str]:
    """Render persistent per-space user preferences stored in memory store."""
    ws_memory = request_context.get("workspace_memory")
    if not (isinstance(ws_memory, Mapping) and ws_memory):
        return []
    mem_parts = [f"{k}={dict(v)}" for k, v in list(ws_memory.items())[:5] if isinstance(v, Mapping)]
    if not mem_parts:
        return []
    return [f"- Persistent space preferences: {'; '.join(mem_parts)}"]


DEFAULT_CONTEXT_SECTIONS: tuple[ContextSectionRenderer, ...] = (
    render_attached_sources,
    render_project_context,
    render_editor_context,
    render_external_scope,
    render_space_memory,
)


def render_request_context(
    request_context: Mapping[str, Any] | None,
    *,
    sections: Sequence[ContextSectionRenderer] = DEFAULT_CONTEXT_SECTIONS,
) -> str:
    """Format active request context into the Supervisor `## Active Request Context` block."""
    if not request_context:
        return ""
    lines: list[str] = []
    for section in sections:
        lines.extend(section(request_context))
    if not lines:
        return ""
    return "## Active Request Context\n" + "\n".join(lines) + "\n\n"


__all__ = [
    "DEFAULT_CONTEXT_SECTIONS",
    "ContextSectionRenderer",
    "format_attached_source_line",
    "render_attached_sources",
    "render_editor_context",
    "render_external_scope",
    "render_project_context",
    "render_request_context",
    "render_space_memory",
]
