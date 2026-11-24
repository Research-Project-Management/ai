"""Deterministic anchor search/replace resolution with ordered fallback matchers."""

from __future__ import annotations

import re
import uuid
import zlib
from collections.abc import Sequence
from typing import Protocol

from ai.agents.editor.context.latex_scanner import normalize_newlines
from ai.agents.editor.patching.contracts import EditorPatchProposal, MatchStrategy, PatchKind

_LINE_PREFIX_RE = re.compile(r"^L\d{2,}:\s", re.MULTILINE)
_BEGIN_DOCUMENT_RE = re.compile(r"^[ \t]*\\begin\{document\}", re.MULTILINE)
_INLINE_WS_RE = re.compile(r"[ \t]+")


def _collapse_line_ws(line: str) -> str:
    return _INLINE_WS_RE.sub(" ", line.rstrip("\r\n").strip())


class AnchorStrategy(Protocol):
    """Strategy interface locating candidate `(start_idx, end_idx)` spans in normalized text."""

    name: MatchStrategy

    def locate(self, content: str, search_text: str) -> list[tuple[int, int]]: ...


class ExactAnchorStrategy:
    """First-pass strategy matching verbatim byte/character substrings."""

    name: MatchStrategy = "exact"

    def locate(self, content: str, search_text: str) -> list[tuple[int, int]]:
        positions: list[tuple[int, int]] = []
        if not search_text:
            return positions
        start = 0
        while True:
            idx = content.find(search_text, start)
            if idx == -1:
                break
            positions.append((idx, idx + len(search_text)))
            start = idx + 1
        return positions


class TrimmedLineAnchorStrategy:
    """Fallback strategy matching multi-line blocks after edge and inline whitespace normalization."""

    name: MatchStrategy = "normalized_whitespace"

    def locate(self, content: str, search_text: str) -> list[tuple[int, int]]:
        search_lines = [_collapse_line_ws(ln) for ln in search_text.splitlines()]
        while search_lines and search_lines[0] == "":
            search_lines.pop(0)
        while search_lines and search_lines[-1] == "":
            search_lines.pop()
        if not search_lines:
            return []

        raw_lines = content.splitlines(keepends=True)
        if len(raw_lines) < len(search_lines):
            return []

        line_starts: list[int] = []
        offset = 0
        for ln in raw_lines:
            line_starts.append(offset)
            offset += len(ln)

        stripped_doc = [_collapse_line_ws(ln) for ln in raw_lines]
        window = len(search_lines)
        matches: list[tuple[int, int]] = []

        # O(N + M) hash pre-filter on first/last anchor lines before slice equality
        first_hash = zlib.crc32(search_lines[0].encode("utf-8"))
        last_hash = zlib.crc32(search_lines[-1].encode("utf-8"))
        doc_hashes = [zlib.crc32(ln.encode("utf-8")) for ln in stripped_doc]

        for idx in range(len(raw_lines) - window + 1):
            if (
                doc_hashes[idx] == first_hash
                and doc_hashes[idx + window - 1] == last_hash
                and stripped_doc[idx : idx + window] == search_lines
            ):
                start_off = line_starts[idx]
                last_raw = raw_lines[idx + window - 1].rstrip("\r\n")
                end_off = line_starts[idx + window - 1] + len(last_raw)
                matches.append((start_off, end_off))
        return matches


DEFAULT_ANCHOR_STRATEGIES: tuple[AnchorStrategy, ...] = (
    ExactAnchorStrategy(),
    TrimmedLineAnchorStrategy(),
)


def _line_span(text: str, start_idx: int, end_idx: int) -> tuple[int, int]:
    from_line = text.count("\n", 0, start_idx) + 1
    trimmed_end = (
        max(start_idx, end_idx - 1)
        if end_idx > start_idx and text[end_idx - 1] == "\n"
        else end_idx
    )
    to_line = max(from_line, text.count("\n", 0, trimmed_end) + 1)
    return from_line, to_line


def offset_span_to_line_col(
    content: str,
    start_idx: int,
    end_idx: int,
) -> tuple[int, int, int, int]:
    """Convert `[start_idx, end_idx)` to `(from_line, from_col, to_line, to_col)` (1-based line, 0-based col)."""
    from_line = content.count("\n", 0, start_idx) + 1
    from_nl = content.rfind("\n", 0, start_idx)
    from_col = start_idx - (from_nl + 1)

    to_line = content.count("\n", 0, end_idx) + 1
    to_nl = content.rfind("\n", 0, end_idx)
    to_col = end_idx - (to_nl + 1)
    return from_line, from_col, to_line, to_col


def find_unique_anchor(
    content: str,
    search_text: str,
    *,
    file_path: str = "document.tex",
    strategies: Sequence[AnchorStrategy] = DEFAULT_ANCHOR_STRATEGIES,
) -> tuple[tuple[int, int] | None, MatchStrategy, str | None]:
    """Locate one unambiguous anchor span `(start_idx, end_idx)` via strategy chain."""
    norm_content = normalize_newlines(content)
    norm_search = normalize_newlines(search_text)

    if not norm_content:
        return (
            None,
            "exact",
            (
                f"Target file '{file_path}' is empty or not loaded. Read the file first with `read_file`."
            ),
        )

    if not norm_search.strip():
        return (
            None,
            "exact",
            (
                "`search_text` is required to anchor this edit. Provide 2-5 verbatim lines from the file."
            ),
        )

    for strategy in strategies:
        matches = strategy.locate(norm_content, norm_search)
        if len(matches) == 1:
            return matches[0], strategy.name, None
        if len(matches) > 1:
            lines = [_line_span(norm_content, s, e)[0] for s, e in matches[:5]]
            qualifier = "" if strategy.name == "exact" else f" ({strategy.name})"
            return (
                None,
                strategy.name,
                (
                    f"`search_text` matched {len(matches)} places in '{file_path}' "
                    f"(lines {lines}){qualifier}. Include additional surrounding lines so the match is unique."
                ),
            )

    prefix_hint = ""
    if _LINE_PREFIX_RE.search(norm_search):
        prefix_hint = " Remove any 'L###: ' line-number prefixes from `search_text`."
    return (
        None,
        "exact",
        (
            f"`search_text` was not found verbatim in '{file_path}'.{prefix_hint} "
            "Call `read_file` to inspect exact lines before retrying."
        ),
    )


def _resolve_preamble_patch(
    norm_content: str,
    norm_replace: str,
    *,
    pid: str,
    file_id: str | None,
    file_path: str,
    clean_title: str,
    rationale: str,
) -> tuple[EditorPatchProposal | None, str, str | None]:
    clean_addition = norm_replace.strip()
    if not clean_addition:
        return (
            None,
            norm_content,
            "insert_preamble requires non-empty `replace_text` (e.g. `\\usepackage{booktabs}`).",
        )
    match = _BEGIN_DOCUMENT_RE.search(norm_content)
    if not match:
        return (
            None,
            norm_content,
            (
                f"`\\begin{{document}}` was not found in '{file_path}'. "
                "Use `replace` on `main.tex` or the file defining the preamble."
            ),
        )
    preamble_slice = norm_content[: match.start()]
    if clean_addition in preamble_slice:
        return (
            None,
            norm_content,
            f"`{clean_addition}` is already present in the preamble of '{file_path}'.",
        )

    anchor = match.group(0)
    start_idx, end_idx = match.start(), match.end()
    effective_replace = f"{clean_addition}\n{anchor}"
    patched = norm_content[:start_idx] + effective_replace + norm_content[end_idx:]
    from_line, to_line = _line_span(norm_content, start_idx, end_idx)
    return (
        EditorPatchProposal(
            id=pid,
            file_id=file_id,
            file_path=file_path,
            kind="insert_preamble",
            title=clean_title,
            rationale=rationale.strip(),
            search_text=anchor,
            replace_text=effective_replace,
            from_line=from_line,
            to_line=to_line,
            match_strategy="preamble",
        ),
        patched,
        None,
    )


def resolve_patch(
    content: str,
    *,
    file_id: str | None,
    file_path: str,
    kind: PatchKind,
    search_text: str,
    replace_text: str,
    title: str,
    rationale: str = "",
    patch_id: str = "",
) -> tuple[EditorPatchProposal | None, str, str | None]:
    """Resolve an edit against `content` and return `(proposal, patched_content, error)`."""
    norm_content = normalize_newlines(content)
    norm_search = normalize_newlines(search_text)
    norm_replace = normalize_newlines(replace_text)
    clean_title = title.strip() or "LaTeX Edit"
    pid = patch_id.strip() or f"patch-{uuid.uuid4().hex[:10]}"

    if not norm_content:
        return (
            None,
            norm_content,
            (
                f"Target file '{file_path}' is empty or not loaded. Read the file first with `read_file`."
            ),
        )

    if kind == "insert_preamble":
        return _resolve_preamble_patch(
            norm_content,
            norm_replace,
            pid=pid,
            file_id=file_id,
            file_path=file_path,
            clean_title=clean_title,
            rationale=rationale,
        )

    chosen_span, strategy, anchor_err = find_unique_anchor(
        norm_content,
        norm_search,
        file_path=file_path,
    )
    if anchor_err or chosen_span is None:
        return None, norm_content, anchor_err or "Anchor resolution failed."

    start_idx, end_idx = chosen_span
    canonical_search = norm_content[start_idx:end_idx]

    if kind == "insert_after":
        sep = (
            "\n"
            if canonical_search
            and not canonical_search.endswith("\n")
            and norm_replace
            and not norm_replace.startswith("\n")
            else ""
        )
        effective_replace = canonical_search + sep + norm_replace
    elif kind == "insert_before":
        sep = (
            "\n"
            if norm_replace
            and not norm_replace.endswith("\n")
            and canonical_search
            and not canonical_search.startswith("\n")
            else ""
        )
        effective_replace = norm_replace + sep + canonical_search
    else:
        effective_replace = norm_replace

    if effective_replace == canonical_search:
        return None, norm_content, "No-op edit: replacement is identical to existing text."

    patched = norm_content[:start_idx] + effective_replace + norm_content[end_idx:]
    from_line, to_line = _line_span(norm_content, start_idx, end_idx)

    return (
        EditorPatchProposal(
            id=pid,
            file_id=file_id,
            file_path=file_path,
            kind=kind,
            title=clean_title,
            rationale=rationale.strip(),
            search_text=canonical_search,
            replace_text=effective_replace,
            from_line=from_line,
            to_line=to_line,
            match_strategy=strategy,
        ),
        patched,
        None,
    )


def check_patch_overlap(
    existing: Sequence[EditorPatchProposal],
    candidate: EditorPatchProposal,
) -> str | None:
    """Reject overlapping hunks targeting the same file within one turn."""
    cand_key = (candidate.file_id or "").strip() or candidate.file_path.strip()
    for prior in existing:
        prior_key = (prior.file_id or "").strip() or prior.file_path.strip()
        if prior_key != cand_key:
            continue
        if prior.from_line <= candidate.to_line and candidate.from_line <= prior.to_line:
            return (
                f"Overlapping patch rejected on '{candidate.file_path}' "
                f"(L{candidate.from_line}-L{candidate.to_line} overlaps with prior '{prior.title}' "
                f"at L{prior.from_line}-L{prior.to_line}). "
                "Combine edits in the same region into one `propose_edit` call."
            )
    return None
