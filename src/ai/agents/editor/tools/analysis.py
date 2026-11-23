"""Read-only citation, lint, review comment, and multi-file structure analysis tools (`Tier 0`)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ai.agents.editor.context.citation_audit import audit_citation_gaps
from ai.agents.editor.context.preamble import extract_local_symbols
from ai.agents.editor.context.structure_audit import audit_page_structure
from ai.agents.editor.gateway.contracts import EditorGatewayError
from ai.agents.editor.tools.contracts import (
    EditorToolResult,
    EditorToolRuntime,
    EditorToolSpec,
)
from ai.agents.editor.tools.file_ops import EmptyArgs, resolve_and_cache_file


class InspectCitationsArgs(BaseModel):
    query: str = Field(default="", description="Optional author, title, key, or year filter.")
    limit: int = Field(default=30, ge=1, le=100)


class LintPagesArgs(BaseModel):
    file_path: str | None = Field(
        default=None,
        description="Specific cached/open file path to lint; defaults to active file or whole project.",
    )
    whole_project: bool = Field(
        default=False,
        description="Set true to run project-wide LaTeX + cross-file citation/label lint.",
    )


class ListReviewCommentsArgs(BaseModel):
    file_id: str | None = Field(default=None, description="Optional target document UUID.")
    file_path: str | None = Field(
        default=None,
        description="Optional file path (defaults to active file).",
    )
    status: Literal["open", "resolved", "all"] = Field(
        default="open",
        description="Filter review comments by status.",
    )
    include_suggestions: bool = Field(
        default=True,
        description="Also include pending Track Changes suggestions for the file.",
    )


class ScanCitationGapsArgs(BaseModel):
    file_path: str | None = Field(
        default=None,
        description="Target .tex file path to scan (defaults to active file).",
    )
    file_id: str | None = Field(default=None, description="Optional target file UUID.")
    max_candidates: int = Field(
        default=15,
        ge=1,
        le=30,
        description="Maximum uncited claim candidates to return.",
    )


class AuditPageStructureArgs(BaseModel):
    include_unreferenced_labels: bool = Field(
        default=False,
        description="Set true to also list labels defined in the page without any \\ref.",
    )
    max_issues_per_category: int = Field(
        default=25,
        ge=5,
        le=100,
        description="Maximum issues to return per structural/ref/package category.",
    )


def _is_tex_or_sty(clean_path: str, *, is_root: bool = False) -> bool:
    lower = clean_path.lower()
    if lower.endswith(".bib"):
        return False
    filename = clean_path.rsplit("/", 1)[-1]
    return is_root or lower.endswith((".tex", ".sty")) or "." not in filename


async def handle_inspect_citations(
    runtime: EditorToolRuntime,
    args: InspectCitationsArgs,
) -> EditorToolResult:
    entries = await runtime.gateway.list_citations(
        runtime.scope,
        query=args.query,
        limit=args.limit,
    )
    for entry in entries:
        runtime.known_cite_keys.add(entry.key)
    return EditorToolResult(
        tool="inspect_citations",
        status="ok",
        data={
            "total": len(entries),
            "entries": [
                {
                    "key": e.key,
                    "entry_type": e.entry_type,
                    "title": e.title,
                    "authors": e.authors,
                    "year": e.year,
                    "file_path": e.file_path,
                }
                for e in entries
            ],
        },
    )


async def handle_validate_citations(
    runtime: EditorToolRuntime,
    args: EmptyArgs,
) -> EditorToolResult:
    del args
    report = await runtime.gateway.validate_citations(runtime.scope)
    return EditorToolResult(tool="validate_citations", status="ok", data=report)


async def handle_lint_pages(
    runtime: EditorToolRuntime,
    args: LintPagesArgs,
) -> EditorToolResult:
    target_path = args.file_path or runtime.scope.active_file_path
    cached = None if args.whole_project else runtime.get_cached_file(file_path=target_path)
    if cached is not None and cached.content:
        report = await runtime.gateway.lint_pages(
            runtime.scope,
            content=cached.content,
            file_path=cached.file_path,
        )
    else:
        report = await runtime.gateway.lint_pages(runtime.scope)
    return EditorToolResult(tool="lint_pages", status="ok", data=report)


async def handle_list_review_comments(
    runtime: EditorToolRuntime,
    args: ListReviewCommentsArgs,
) -> EditorToolResult:
    target_id = (args.file_id or "").strip() or None
    target_path = (args.file_path or "").strip().lstrip("/") or None
    if not target_id and target_path:
        cached = runtime.get_cached_file(file_path=target_path)
        if cached and cached.file_id:
            target_id = cached.file_id
        else:
            files = await runtime.gateway.list_files(runtime.scope, include_content=False)
            exact = [
                f for f in files if f.path.lstrip("/") == target_path or f.title == target_path
            ]
            if exact:
                target_id = exact[0].id
    target_id = target_id or runtime.scope.active_file_id or runtime.scope.page_id or None

    status_filter = None if args.status == "all" else args.status
    comments = await runtime.gateway.list_review_comments(
        runtime.scope,
        doc_id=target_id,
        status=status_filter,
    )
    suggestions = (
        await runtime.gateway.list_review_suggestions(
            runtime.scope,
            doc_id=target_id,
            status="pending",
        )
        if args.include_suggestions
        else []
    )
    return EditorToolResult(
        tool="list_review_comments",
        status="ok",
        data={
            "doc_id": target_id,
            "total_comments": len(comments),
            "comments": [c.model_dump() for c in comments],
            "total_pending_suggestions": len(suggestions),
            "suggestions": [s.model_dump() for s in suggestions],
        },
    )


async def handle_scan_citation_gaps(
    runtime: EditorToolRuntime,
    args: ScanCitationGapsArgs,
) -> EditorToolResult:
    entry, err = await resolve_and_cache_file(
        runtime,
        file_id=args.file_id,
        file_path=args.file_path or runtime.scope.active_file_path,
    )
    if err or entry is None:
        return EditorToolResult(
            tool="scan_citation_gaps",
            status="error",
            error=err or "Could not load file for citation audit.",
        )

    if not runtime.known_cite_keys and runtime.scope.page_id.strip():
        try:
            bib_entries = await runtime.gateway.list_citations(runtime.scope, limit=500)
            for bib in bib_entries:
                runtime.known_cite_keys.add(bib.key)
        except EditorGatewayError:
            pass

    report = audit_citation_gaps(
        entry.content,
        file_path=entry.file_path,
        known_bib_keys=frozenset(runtime.known_cite_keys),
        max_candidates=args.max_candidates,
    )
    return EditorToolResult(
        tool="scan_citation_gaps",
        status="ok",
        data=report.model_dump(mode="json"),
    )


async def handle_audit_page_structure(
    runtime: EditorToolRuntime,
    args: AuditPageStructureArgs,
) -> EditorToolResult:
    tex_files: dict[str, str] = {}
    root_path: str | None = None

    files = []
    if runtime.scope.page_id.strip():
        try:
            files = await runtime.gateway.list_files(runtime.scope, include_content=True)
        except EditorGatewayError:
            files = []

    fallback_reads = 0
    for f in files:
        clean_path = f.path.strip().replace("\\", "/").lstrip("/") or f.title
        if not _is_tex_or_sty(clean_path, is_root=f.is_root):
            continue
        if f.is_root:
            root_path = clean_path

        existing = runtime.get_cached_file(file_id=f.id, file_path=clean_path)
        if f.content is not None:
            tex_files[clean_path] = f.content
            if existing is None:
                runtime.put_cached_file(file_id=f.id, file_path=clean_path, content=f.content)
        elif existing is not None:
            tex_files[clean_path] = existing.content
        elif fallback_reads < 15:
            try:
                doc = await runtime.gateway.read_file(runtime.scope, f.id)
            except EditorGatewayError:
                continue
            fallback_reads += 1
            runtime.put_cached_file(file_id=doc.id, file_path=clean_path, content=doc.content)
            tex_files[clean_path] = doc.content

    for entry in runtime.file_cache.values():
        epath = entry.file_path.strip().replace("\\", "/").lstrip("/")
        if epath and _is_tex_or_sty(epath):
            tex_files[epath] = entry.content

    if not tex_files:
        return EditorToolResult(
            tool="audit_page_structure",
            status="error",
            error="No .tex page files available to audit.",
        )

    for content in tex_files.values():
        local_labels, _ = extract_local_symbols(content)
        runtime.known_labels.update(local_labels)

    report = audit_page_structure(
        tex_files,
        root_file_hint=root_path or runtime.scope.active_file_path,
        include_unreferenced_labels=args.include_unreferenced_labels,
        max_issues_per_category=args.max_issues_per_category,
    )
    return EditorToolResult(
        tool="audit_page_structure",
        status="ok",
        data=report.model_dump(mode="json"),
    )


def build_analysis_tools() -> tuple[EditorToolSpec, ...]:
    """Build Tier-0 citation, lint, review comment, and multi-file structure tools."""
    return (
        EditorToolSpec(
            name="inspect_citations",
            description="List or filter BibTeX entries defined in the project's .bib files.",
            args_schema=InspectCitationsArgs,
            risk_tier="read",
            idempotent=True,
            handler=handle_inspect_citations,
        ),
        EditorToolSpec(
            name="validate_citations",
            description="Check project .bib files for duplicate citation keys or missing metadata.",
            args_schema=EmptyArgs,
            risk_tier="read",
            idempotent=True,
            handler=handle_validate_citations,
        ),
        EditorToolSpec(
            name="lint_pages",
            description="Run static LaTeX linter on a cached file or the whole project.",
            args_schema=LintPagesArgs,
            risk_tier="read",
            idempotent=True,
            handler=handle_lint_pages,
        ),
        EditorToolSpec(
            name="list_review_comments",
            description=(
                "Inspect collaborator review comment threads and pending Track Changes suggestions "
                "on a page file."
            ),
            args_schema=ListReviewCommentsArgs,
            risk_tier="read",
            idempotent=True,
            handler=handle_list_review_comments,
        ),
        EditorToolSpec(
            name="scan_citation_gaps",
            description=(
                "Citation Reviewer: scan a .tex file for undefined \\cite keys, unused .bib entries, "
                "and claims/benchmarks likely needing citations."
            ),
            args_schema=ScanCitationGapsArgs,
            risk_tier="read",
            idempotent=True,
            handler=handle_scan_citation_gaps,
        ),
        EditorToolSpec(
            name="audit_page_structure",
            description=(
                "Structure Auditor: analyze multi-file outline balance, reading order, word counts, "
                "duplicate/undefined \\label & \\ref links, \\input graph, orphan .tex files, and "
                "missing packages."
            ),
            args_schema=AuditPageStructureArgs,
            risk_tier="read",
            idempotent=True,
            handler=handle_audit_page_structure,
        ),
    )
