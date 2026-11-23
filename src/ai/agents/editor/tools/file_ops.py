"""Read-only page navigation, file slicing, symbol & full-text search tools (`Tier 0`)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ai.agents.editor.tools.contracts import (
    CachedFile,
    EditorToolResult,
    EditorToolRuntime,
    EditorToolSpec,
)

_MAX_READ_LINES_PER_CALL = 250


class ListFilesArgs(BaseModel):
    include_content: bool = Field(
        default=False,
        description="Set true only if you need a preview of small projects; usually false.",
    )


class ReadFileArgs(BaseModel):
    file_id: str | None = Field(default=None, description="Optional target file UUID.")
    file_path: str | None = Field(
        default=None,
        description="Relative file path (e.g. 'sections/intro.tex' or 'references.bib').",
    )
    start_line: int = Field(default=1, ge=1, description="1-based first line to inspect.")
    end_line: int | None = Field(
        default=None,
        ge=1,
        description="1-based last line to inspect (capped at 250 lines per call).",
    )


class EmptyArgs(BaseModel):
    pass


class SearchPagesArgs(BaseModel):
    query: str = Field(..., min_length=1, description="Text or regex pattern to find.")
    case_sensitive: bool = False
    whole_word: bool = False
    use_regex: bool = False
    max_results: int = Field(default=20, ge=1, le=50)


async def resolve_and_cache_file(
    runtime: EditorToolRuntime,
    *,
    file_id: str | None = None,
    file_path: str | None = None,
) -> tuple[CachedFile | None, str | None]:
    """Resolve `file_id` or `file_path` via turn cache first, falling back to `EditorGateway`."""
    target_id = (file_id or "").strip() or None
    target_path = (file_path or "").strip().lstrip("/") or None

    cached = runtime.get_cached_file(file_id=target_id, file_path=target_path)
    if cached is not None:
        return cached, None

    if not target_id and target_path:
        files = await runtime.gateway.list_files(runtime.scope, include_content=False)
        exact = [f for f in files if f.path.lstrip("/") == target_path or f.title == target_path]
        if len(exact) == 1:
            target_id = exact[0].id
        else:
            basename = target_path.rsplit("/", 1)[-1]
            by_name = [f for f in files if f.path.rsplit("/", 1)[-1] == basename]
            if len(by_name) == 1:
                target_id = by_name[0].id
            elif len(by_name) > 1:
                candidates = ", ".join(f.path for f in by_name[:5])
                return None, (
                    f"Ambiguous file path '{target_path}'. Specify full path from: {candidates}"
                )
            else:
                available = ", ".join(f.path for f in files[:15]) or "(none)"
                return None, (
                    f"File '{target_path}' not found in project. Available files: {available}"
                )

    if not target_id:
        target_id = runtime.scope.active_file_id or runtime.scope.page_id
    if not target_id:
        return None, "Specify `file_path` or `file_id` to read a file."

    doc = await runtime.gateway.read_file(runtime.scope, target_id)
    entry = runtime.put_cached_file(
        file_id=doc.id,
        file_path=target_path or doc.path,
        content=doc.content,
    )
    return entry, None


def _slice_numbered_content(
    entry: CachedFile,
    *,
    start_line: int,
    end_line: int | None,
) -> dict[str, Any]:
    lines = entry.content.splitlines()
    total_lines = len(lines)
    if total_lines == 0:
        return {
            "file_id": entry.file_id,
            "file_path": entry.file_path,
            "total_lines": 0,
            "start_line": 1,
            "end_line": 0,
            "has_more": False,
            "content": "(empty file)",
        }

    s_line = max(1, min(start_line, total_lines))
    requested_end = end_line if end_line is not None else s_line + _MAX_READ_LINES_PER_CALL - 1
    e_line = min(total_lines, max(s_line, requested_end), s_line + _MAX_READ_LINES_PER_CALL - 1)
    width = max(3, len(str(total_lines)))
    numbered = "\n".join(f"L{idx:0{width}d}: {lines[idx - 1]}" for idx in range(s_line, e_line + 1))
    return {
        "file_id": entry.file_id,
        "file_path": entry.file_path,
        "total_lines": total_lines,
        "start_line": s_line,
        "end_line": e_line,
        "has_more": e_line < total_lines,
        "content": numbered,
    }


async def handle_list_files(runtime: EditorToolRuntime, args: ListFilesArgs) -> EditorToolResult:
    files = await runtime.gateway.list_files(runtime.scope, include_content=args.include_content)
    items: list[dict[str, Any]] = []
    for f in files[:60]:
        if f.content is not None:
            runtime.put_cached_file(file_id=f.id, file_path=f.path, content=f.content)
        items.append(
            {
                "id": f.id,
                "path": f.path,
                "title": f.title,
                "is_root": f.is_root,
                **({"line_count": len(f.content.splitlines())} if f.content is not None else {}),
            }
        )
    return EditorToolResult(
        tool="list_files",
        status="ok",
        data={"total_files": len(files), "files": items},
    )


async def handle_read_file(runtime: EditorToolRuntime, args: ReadFileArgs) -> EditorToolResult:
    entry, err = await resolve_and_cache_file(
        runtime,
        file_id=args.file_id,
        file_path=args.file_path or runtime.scope.active_file_path,
    )
    if err or entry is None:
        return EditorToolResult(
            tool="read_file",
            status="error",
            error=err or "Could not load file",
        )
    return EditorToolResult(
        tool="read_file",
        status="ok",
        data=_slice_numbered_content(
            entry,
            start_line=args.start_line,
            end_line=args.end_line,
        ),
    )


async def handle_get_outline_and_symbols(
    runtime: EditorToolRuntime,
    args: EmptyArgs,
) -> EditorToolResult:
    del args
    catalog = await runtime.gateway.get_symbols(runtime.scope)
    runtime.known_labels.update(catalog.labels)
    runtime.known_cite_keys.update(catalog.citations)
    return EditorToolResult(
        tool="get_outline_and_symbols",
        status="ok",
        data={
            "root_doc_path": catalog.root_doc_path,
            "labels": list(catalog.labels[:60]),
            "citations": list(catalog.citations[:60]),
            "packages": list(catalog.packages[:40]),
            "commands": list(catalog.commands[:40]),
            "environments": list(catalog.environments[:30]),
            "outline": catalog.outline[:50],
            "files": catalog.files[:50],
        },
    )


async def handle_search_pages(
    runtime: EditorToolRuntime,
    args: SearchPagesArgs,
) -> EditorToolResult:
    hits = await runtime.gateway.search_pages(
        runtime.scope,
        query=args.query,
        case_sensitive=args.case_sensitive,
        whole_word=args.whole_word,
        use_regex=args.use_regex,
        max_results=args.max_results,
    )
    return EditorToolResult(
        tool="search_pages",
        status="ok",
        data={
            "query": args.query,
            "total_hits": len(hits),
            "hits": [h.model_dump() for h in hits],
        },
    )


def build_file_tools() -> tuple[EditorToolSpec, ...]:
    """Build Tier-0 file navigation, symbol outline, and page search tools."""
    return (
        EditorToolSpec(
            name="list_files",
            description="List all files and paths in the active LaTeX page project.",
            args_schema=ListFilesArgs,
            risk_tier="read",
            idempotent=True,
            handler=handle_list_files,
        ),
        EditorToolSpec(
            name="read_file",
            description=(
                "Read line-numbered content of any .tex/.bib file in the project "
                "(supports `start_line`/`end_line` pagination)."
            ),
            args_schema=ReadFileArgs,
            risk_tier="read",
            idempotent=True,
            handler=handle_read_file,
        ),
        EditorToolSpec(
            name="get_outline_and_symbols",
            description=(
                "Fetch project-wide outline sections, defined \\label keys, .bib \\cite keys, "
                "loaded packages, and custom macros/environments."
            ),
            args_schema=EmptyArgs,
            risk_tier="read",
            idempotent=True,
            handler=handle_get_outline_and_symbols,
        ),
        EditorToolSpec(
            name="search_pages",
            description="Search across all files in the page by keyword or regex.",
            args_schema=SearchPagesArgs,
            risk_tier="read",
            idempotent=True,
            handler=handle_search_pages,
        ),
    )
