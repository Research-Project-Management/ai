"""HITL-gated bibliography, page creation, and Track Changes write tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ai.agents.editor.context.latex_scanner import normalize_newlines
from ai.agents.editor.gateway.http import editor_page_route
from ai.agents.editor.patching.matcher import find_unique_anchor, offset_span_to_line_col
from ai.agents.editor.tools.contracts import (
    EditorToolResult,
    EditorToolRuntime,
    EditorToolSpec,
)
from ai.agents.editor.tools.file_ops import resolve_and_cache_file
from ai.shared.resilience.mutations import MutationRecord, record_turn_mutation


class AddBibtexEntriesArgs(BaseModel):
    raw_bibtex: str | None = Field(
        default=None,
        description="Raw BibTeX entry/entries verified from Library or peer search results.",
    )
    entries: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Optional structured citation objects if raw BibTeX was not provided.",
    )
    target_bib_file: str = Field(
        default="references.bib",
        description="Target .bib filename in the project.",
    )


class ResolveCitationArgs(BaseModel):
    identifier: str = Field(
        ...,
        min_length=3,
        description="Verified DOI (e.g. 10.1145/...) or arXiv ID (e.g. 1706.03762).",
    )
    target_bib_file: str = Field(
        default="references.bib",
        description="Target .bib filename to append resolved entry into.",
    )


class SyncLibraryCitationsArgs(BaseModel):
    collection_id: str = Field(..., min_length=1, description="Library collection UUID.")
    target_bib_file: str = Field(
        default="references.bib",
        description="Target .bib filename in the project.",
    )


class CreateReviewSuggestionArgs(BaseModel):
    file_path: str | None = Field(
        default=None,
        description="Target page file path (defaults to active file).",
    )
    file_id: str | None = Field(default=None, description="Optional target document UUID.")
    change_type: Literal["insert", "delete"] = Field(
        default="insert",
        description="Track Changes operation: 'insert' after anchor or 'delete' anchor span.",
    )
    search_text: str = Field(
        ...,
        min_length=1,
        description="Verbatim anchor text in the target file.",
    )
    suggested_text: str = Field(
        default="",
        description="Text to insert immediately after `search_text` when `change_type='insert'`.",
    )
    rationale: str = Field(default="", description="Optional note explaining why.")


class CreatePageArgs(BaseModel):
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Title of the new LaTeX Page / academic article.",
    )
    description: str | None = Field(
        default=None,
        description="Optional short abstract or description of the new page.",
    )
    content: str | None = Field(
        default=None,
        description="Optional initial LaTeX source content; omit to scaffold starter LaTeX.",
    )
    template_type: str | None = Field(
        default=None,
        description="Optional backend template key (e.g. 'article', 'blank', 'example').",
    )


def _invalidate_bib_cache(runtime: EditorToolRuntime, target_bib_file: str) -> None:
    clean = target_bib_file.strip().lstrip("/")
    if clean:
        runtime.file_cache.pop(clean, None)


def _record_returned_keys(runtime: EditorToolRuntime, payload: Any) -> list[str]:
    added_keys: list[str] = []
    if not isinstance(payload, dict):
        return added_keys

    single_entry = payload.get("entry")
    if isinstance(single_entry, dict):
        k = str(single_entry.get("key") or single_entry.get("citeKey") or "").strip()
        if k:
            runtime.known_cite_keys.add(k)
            added_keys.append(k)

    raw_entries = payload.get("entries")
    if isinstance(raw_entries, list):
        for item in raw_entries:
            if isinstance(item, dict):
                k = str(item.get("key") or item.get("citeKey") or "").strip()
                if k:
                    runtime.known_cite_keys.add(k)
                    added_keys.append(k)

    direct_key = payload.get("key") or payload.get("citeKey")
    if isinstance(direct_key, str) and direct_key.strip():
        runtime.known_cite_keys.add(direct_key.strip())
        added_keys.append(direct_key.strip())

    return list(dict.fromkeys(added_keys))


async def handle_add_bibtex_entries(
    runtime: EditorToolRuntime,
    args: AddBibtexEntriesArgs,
) -> EditorToolResult:
    has_raw = bool(args.raw_bibtex and args.raw_bibtex.strip())
    if not has_raw and not args.entries:
        return EditorToolResult(
            tool="add_bibtex_entries",
            status="error",
            error="Provide non-empty `raw_bibtex` or structured `entries`.",
        )

    res = await runtime.gateway.add_bibtex_entries(
        runtime.scope,
        raw_bibtex=args.raw_bibtex,
        entries=args.entries,
        target_bib_file=args.target_bib_file,
    )
    _invalidate_bib_cache(runtime, args.target_bib_file)
    keys = _record_returned_keys(runtime, res)
    return EditorToolResult(
        tool="add_bibtex_entries",
        status="ok",
        data={"added_keys": keys, "result": res},
    )


async def handle_resolve_citation(
    runtime: EditorToolRuntime,
    args: ResolveCitationArgs,
) -> EditorToolResult:
    res = await runtime.gateway.resolve_citation(
        runtime.scope,
        identifier=args.identifier,
        target_bib_file=args.target_bib_file,
    )
    _invalidate_bib_cache(runtime, args.target_bib_file)
    keys = _record_returned_keys(runtime, res)
    return EditorToolResult(
        tool="resolve_citation",
        status="ok",
        data={"added_keys": keys, "result": res},
    )


async def handle_sync_library_citations(
    runtime: EditorToolRuntime,
    args: SyncLibraryCitationsArgs,
) -> EditorToolResult:
    res = await runtime.gateway.sync_library_citations(
        runtime.scope,
        collection_id=args.collection_id,
        target_bib_file=args.target_bib_file,
    )
    _invalidate_bib_cache(runtime, args.target_bib_file)
    _record_returned_keys(runtime, res)
    return EditorToolResult(tool="sync_library_citations", status="ok", data=res)


async def handle_create_review_suggestion(
    runtime: EditorToolRuntime,
    args: CreateReviewSuggestionArgs,
) -> EditorToolResult:
    if args.change_type == "insert" and not args.suggested_text:
        return EditorToolResult(
            tool="create_review_suggestion",
            status="error",
            error="`suggested_text` is required when `change_type='insert'`.",
        )

    target_path = (args.file_path or runtime.scope.active_file_path or "main.tex").lstrip("/")
    cached, err = await resolve_and_cache_file(
        runtime,
        file_id=args.file_id,
        file_path=target_path,
    )
    if err or cached is None:
        return EditorToolResult(
            tool="create_review_suggestion",
            status="error",
            error=err or f"Could not load target file '{target_path}'.",
        )

    doc_id = (cached.file_id or runtime.scope.active_file_id or runtime.scope.page_id).strip()
    if not doc_id:
        return EditorToolResult(
            tool="create_review_suggestion",
            status="error",
            error=f"Cannot create suggestion: missing document ID for '{cached.file_path}'.",
        )

    norm_content = normalize_newlines(cached.content)
    span, _strategy, anchor_err = find_unique_anchor(
        norm_content,
        args.search_text,
        file_path=cached.file_path,
    )
    if anchor_err or span is None:
        return EditorToolResult(
            tool="create_review_suggestion",
            status="error",
            error=anchor_err or "Anchor resolution failed.",
        )

    start_idx, end_idx = span
    canonical_search = norm_content[start_idx:end_idx]
    from_line, from_col, to_line, to_col = offset_span_to_line_col(norm_content, start_idx, end_idx)

    if args.change_type == "insert":
        res = await runtime.gateway.create_review_suggestion(
            runtime.scope,
            doc_id=doc_id,
            change_type="insert",
            suggested_text=args.suggested_text,
            original_text="",
            from_line=to_line,
            from_column=to_col,
            to_line=to_line,
            to_column=to_col,
        )
    else:
        res = await runtime.gateway.create_review_suggestion(
            runtime.scope,
            doc_id=doc_id,
            change_type="delete",
            suggested_text="",
            original_text=canonical_search,
            from_line=from_line,
            from_column=from_col,
            to_line=to_line,
            to_column=to_col,
        )

    return EditorToolResult(tool="create_review_suggestion", status="ok", data=res)


async def handle_create_page(
    runtime: EditorToolRuntime,
    args: CreatePageArgs,
) -> EditorToolResult:
    created = await runtime.gateway.create_page(
        runtime.scope,
        title=args.title,
        description=args.description,
        content=args.content,
        template_type=args.template_type,
    )
    page_id = str(created.get("id") or created.get("pageId") or "").strip()
    if page_id:
        record_turn_mutation(
            MutationRecord(
                tool="create_page",
                entity_type="page",
                entity_id=page_id,
                label=args.title,
                undo_method="DELETE",
                undo_path=editor_page_route(page_id),
            )
        )
    return EditorToolResult(tool="create_page", status="ok", data=created)


def _confirm_add_bibtex(args: dict[str, Any]) -> str:
    target_bib = str(args.get("target_bib_file") or "references.bib")
    return f"Add BibTeX citation entries to `{target_bib}`"


def _confirm_resolve_citation(args: dict[str, Any]) -> str:
    ident = str(args.get("identifier") or "")
    target_bib = str(args.get("target_bib_file") or "references.bib")
    return f"Resolve academic identifier `{ident}` and append BibTeX to `{target_bib}`"


def _confirm_sync_library(args: dict[str, Any]) -> str:
    col_id = str(args.get("collection_id") or "")
    target_bib = str(args.get("target_bib_file") or "references.bib")
    return f"Sync Library collection `{col_id}` into `{target_bib}`"


def _confirm_create_suggestion(args: dict[str, Any]) -> str:
    op = str(args.get("change_type") or "insert")
    fpath = str(args.get("file_path") or "active page file")
    return f"Create Track Changes `{op}` suggestion in `{fpath}`"


def _confirm_create_page(args: dict[str, Any]) -> str:
    title = str(args.get("title") or "Untitled Page")
    return f"Create new Page `{title}`"


def build_write_tools() -> tuple[EditorToolSpec, ...]:
    """Build HITL-gated page, bibliography, and Track Changes write tools."""
    return (
        EditorToolSpec(
            name="create_page",
            description=(
                "HITL-gated write: create/initialize a new LaTeX Page / academic article in the active "
                "workspace/project (works even when no existing page is currently open)."
            ),
            args_schema=CreatePageArgs,
            risk_tier="write",
            idempotent=False,
            handler=handle_create_page,
            describe_confirmation=_confirm_create_page,
        ),
        EditorToolSpec(
            name="add_bibtex_entries",
            description=(
                "HITL-gated write: format and append verified BibTeX entries into the project's .bib file."
            ),
            args_schema=AddBibtexEntriesArgs,
            risk_tier="write",
            idempotent=False,
            handler=handle_add_bibtex_entries,
            describe_confirmation=_confirm_add_bibtex,
        ),
        EditorToolSpec(
            name="resolve_citation",
            description=(
                "HITL-gated write: resolve a verified DOI or arXiv ID via Crossref/arXiv and append BibTeX."
            ),
            args_schema=ResolveCitationArgs,
            risk_tier="write",
            idempotent=False,
            handler=handle_resolve_citation,
            describe_confirmation=_confirm_resolve_citation,
        ),
        EditorToolSpec(
            name="sync_library_citations",
            description=(
                "HITL-gated write: sync a Flux Library collection into the project's references .bib file."
            ),
            args_schema=SyncLibraryCitationsArgs,
            risk_tier="write",
            idempotent=False,
            handler=handle_sync_library_citations,
            describe_confirmation=_confirm_sync_library,
        ),
        EditorToolSpec(
            name="create_review_suggestion",
            description=(
                "HITL-gated write: create a Track Changes insert/delete suggestion in the page "
                "(persists cleanly in Review panel without mutating live text until accepted)."
            ),
            args_schema=CreateReviewSuggestionArgs,
            risk_tier="write",
            idempotent=False,
            handler=handle_create_review_suggestion,
            describe_confirmation=_confirm_create_suggestion,
        ),
    )
