"""Backend payload -> Editor DTO mapping helpers."""

from __future__ import annotations

import json
from typing import Any

from ai.agents.editor.gateway.contracts import (
    EditorCitationEntry,
    EditorDocument,
    EditorFileInfo,
    EditorReviewComment,
    EditorReviewSuggestion,
    EditorSearchHit,
    EditorSymbolCatalog,
)


def coerce_content_text(raw_content: Any, raw_lines: Any = None) -> str:
    """Normalize line arrays, JSON-encoded line strings, or raw text into document content."""
    if isinstance(raw_lines, list):
        return "\n".join(str(line) for line in raw_lines)
    if isinstance(raw_content, str):
        stripped = raw_content.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                    return "\n".join(parsed)
            except ValueError:
                pass
        return raw_content
    if isinstance(raw_content, list):
        return "\n".join(str(item) for item in raw_content)
    return ""


def extract_symbol_names(entries: Any, *, keys: tuple[str, ...]) -> tuple[str, ...]:
    """Deduplicate non-empty LaTeX symbol identifiers from mixed string/dict entries."""
    if not isinstance(entries, list):
        return ()
    seen: dict[str, None] = {}
    for entry in entries:
        if isinstance(entry, str) and entry.strip():
            seen.setdefault(entry.strip(), None)
        elif isinstance(entry, dict):
            for key in keys:
                val = entry.get(key)
                if isinstance(val, str) and val.strip():
                    seen.setdefault(val.strip(), None)
                    break
    return tuple(seen.keys())


def extract_author_name(item: dict[str, Any]) -> str:
    """Extract display author name from nested or flat comment/suggestion payloads."""
    author_obj = item.get("author")
    if isinstance(author_obj, dict):
        return str(author_obj.get("name") or "").strip()
    return str(item.get("authorName") or item.get("author") or "").strip()


def build_search_hit(
    match: dict[str, Any],
    *,
    file_id: str = "",
    file_path: str = "",
) -> EditorSearchHit:
    """Project a backend page search match object into an EditorSearchHit."""
    raw_col = match.get("matchStart")
    col = (
        raw_col
        if isinstance(raw_col, int)
        else (match["column"] if isinstance(match.get("column"), int) else None)
    )
    resolved_path = (
        file_path
        or str(match.get("fileName") or match.get("filePath") or "document.tex").lstrip("/")
    )
    preview = str(match.get("text") or match.get("preview") or match.get("snippet") or "")[:300]
    return EditorSearchHit(
        file_id=file_id or str(match.get("fileId") or ""),
        file_path=resolved_path,
        line=max(1, int(match.get("line") or 1)),
        column=col,
        preview=preview,
    )


def project_file_list(raw: Any, *, include_content: bool) -> list[EditorFileInfo]:
    """Map `/docs/{page_id}/files` response payload into EditorFileInfo DTOs."""
    entries = raw.get("files", []) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        return []

    files: list[EditorFileInfo] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        file_id = str(entry.get("id") or entry.get("docId") or "").strip()
        raw_path = str(
            entry.get("path") or entry.get("name") or entry.get("title") or ""
        ).lstrip("/")
        title = str(entry.get("title") or entry.get("name") or raw_path).strip()
        if not raw_path and title:
            raw_path = title
        content = (
            coerce_content_text(entry.get("content"), entry.get("lines"))
            if include_content
            and (entry.get("content") is not None or entry.get("lines") is not None)
            else None
        )
        files.append(
            EditorFileInfo(
                id=file_id,
                path=raw_path or "main.tex",
                title=title or raw_path or "main.tex",
                is_root=bool(
                    entry.get("isRootDoc") or entry.get("isRoot") or entry.get("isMain")
                ),
                content=content,
            )
        )
    return files


def project_document(raw: dict[str, Any], *, doc_id: str) -> EditorDocument:
    """Map `/pages/{doc_id}` JSON payload into an EditorDocument DTO."""
    raw_page = raw.get("page")
    page_obj: dict[str, Any] = raw_page if isinstance(raw_page, dict) else {}
    content = coerce_content_text(
        raw.get("content") if raw.get("content") is not None else page_obj.get("content"),
        raw.get("lines"),
    )
    raw_path = str(
        raw.get("path")
        or raw.get("lastKnownPath")
        or page_obj.get("path")
        or page_obj.get("title")
        or raw.get("title")
        or "document.tex"
    ).lstrip("/")
    title = str(page_obj.get("title") or raw.get("title") or raw_path).strip()
    rev_val = raw.get("rev") if isinstance(raw.get("rev"), int) else raw.get("version")
    rev = int(rev_val) if isinstance(rev_val, int) else None
    line_count = len(content.splitlines()) if content else 0

    return EditorDocument(
        id=str(raw.get("id") or page_obj.get("id") or doc_id),
        path=raw_path or "document.tex",
        title=title or raw_path or "document.tex",
        content=content,
        rev=rev,
        line_count=line_count,
    )


def project_symbol_catalog(raw: Any) -> EditorSymbolCatalog:
    """Map `/structure/symbols` JSON payload into an EditorSymbolCatalog DTO."""
    if not isinstance(raw, dict):
        return EditorSymbolCatalog()

    outline_raw = raw.get("outline")
    outline = (
        [item for item in outline_raw if isinstance(item, dict)]
        if isinstance(outline_raw, list)
        else []
    )
    files_raw = raw.get("files")
    files: list[dict[str, Any]] = []
    if isinstance(files_raw, list):
        for f in files_raw:
            if isinstance(f, dict):
                files.append(f)
            elif isinstance(f, str) and f.strip():
                files.append({"path": f.strip().lstrip("/")})

    return EditorSymbolCatalog(
        root_doc_path=str(raw["rootDocPath"]).lstrip("/") if raw.get("rootDocPath") else None,
        labels=extract_symbol_names(raw.get("labels"), keys=("key", "name", "label")),
        citations=extract_symbol_names(raw.get("citations"), keys=("key", "citeKey", "name")),
        packages=extract_symbol_names(raw.get("packages"), keys=("name", "package")),
        commands=extract_symbol_names(
            raw.get("customCommands") or raw.get("commands"),
            keys=("name", "signature", "command"),
        ),
        environments=extract_symbol_names(
            raw.get("customEnvironments") or raw.get("environments"),
            keys=("name", "environment"),
        ),
        outline=outline[:120],
        files=files[:100],
    )


def project_search_hits(raw: Any, *, max_results: int) -> list[EditorSearchHit]:
    """Map grouped or flat page search payloads into capped EditorSearchHit DTOs."""
    if not isinstance(raw, dict):
        return []

    hits: list[EditorSearchHit] = []
    capped = max(1, min(max_results, 100))

    file_results = raw.get("results") or raw.get("files")
    if isinstance(file_results, list):
        for group in file_results:
            if not isinstance(group, dict):
                continue
            if "matches" in group and isinstance(group["matches"], list):
                fid = str(group.get("fileId") or "")
                fpath = str(group.get("fileName") or group.get("path") or "document.tex").lstrip(
                    "/"
                )
                for m in group["matches"]:
                    if not isinstance(m, dict):
                        continue
                    hits.append(build_search_hit(m, file_id=fid, file_path=fpath))
                    if len(hits) >= capped:
                        return hits
            elif "line" in group:
                hits.append(build_search_hit(group))
                if len(hits) >= capped:
                    return hits

    flat_matches = raw.get("matches")
    if not hits and isinstance(flat_matches, list):
        for m in flat_matches[:capped]:
            if isinstance(m, dict):
                hits.append(build_search_hit(m))
    return hits


def project_citation_entries(raw: Any, *, limit: int) -> list[EditorCitationEntry]:
    """Map `/citations` or `/citations/search` JSON payloads into EditorCitationEntry DTOs."""
    raw_list = raw.get("entries") or raw.get("items") or [] if isinstance(raw, dict) else raw
    if not isinstance(raw_list, list):
        return []

    capped = max(1, min(limit, 500))
    entries: list[EditorCitationEntry] = []
    for item in raw_list[:capped]:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or item.get("citeKey") or "").strip()
        if not key:
            continue
        entries.append(
            EditorCitationEntry(
                key=key,
                entry_type=str(item.get("entryType") or item.get("type") or ""),
                title=str(item.get("title") or "").strip(),
                authors=str(item.get("author") or item.get("authors") or "").strip(),
                year=str(item.get("year") or "").strip(),
                file_path=str(item.get("file") or item.get("filePath") or "").lstrip("/"),
                raw=str(item.get("raw") or item.get("rawBibtex") or "").strip()[:1200],
            )
        )
    return entries


def project_review_comments(raw: Any, *, limit: int = 80) -> list[EditorReviewComment]:
    """Map `/comments` JSON payload into EditorReviewComment DTOs."""
    entries = raw.get("comments", []) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        return []

    comments: list[EditorReviewComment] = []
    for item in entries[:limit]:
        if not isinstance(item, dict):
            continue
        raw_replies = item.get("replies")
        replies = (
            tuple(r for r in raw_replies if isinstance(r, dict))
            if isinstance(raw_replies, list)
            else ()
        )
        comments.append(
            EditorReviewComment(
                id=str(item.get("id") or ""),
                status=str(
                    item.get("status") or ("resolved" if item.get("resolved") else "open")
                ),
                author=extract_author_name(item),
                content=str(item.get("content") or item.get("text") or "").strip()[:1000],
                line=int(item["line"]) if isinstance(item.get("line"), int) else None,
                line_end=int(item["lineEnd"]) if isinstance(item.get("lineEnd"), int) else None,
                replies=replies[:10],
            )
        )
    return comments


def project_review_suggestions(raw: Any, *, limit: int = 80) -> list[EditorReviewSuggestion]:
    """Map `/suggestions` JSON payload into EditorReviewSuggestion DTOs."""
    entries = raw.get("suggestions", []) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        return []

    suggestions: list[EditorReviewSuggestion] = []
    for item in entries[:limit]:
        if not isinstance(item, dict):
            continue
        suggestions.append(
            EditorReviewSuggestion(
                id=str(item.get("id") or ""),
                status=str(item.get("status") or "pending"),
                change_type=str(item.get("type") or item.get("changeType") or "insert"),
                author=extract_author_name(item),
                original_text=str(item.get("originalText") or ""),
                suggested_text=str(item.get("suggestedText") or item.get("text") or ""),
                from_line=int(item["fromLine"]) if isinstance(item.get("fromLine"), int) else None,
                to_line=int(item["toLine"]) if isinstance(item.get("toLine"), int) else None,
            )
        )
    return suggestions


__all__ = [
    "build_search_hit",
    "coerce_content_text",
    "extract_author_name",
    "extract_symbol_names",
    "project_citation_entries",
    "project_document",
    "project_file_list",
    "project_review_comments",
    "project_review_suggestions",
    "project_search_hits",
    "project_symbol_catalog",
]
