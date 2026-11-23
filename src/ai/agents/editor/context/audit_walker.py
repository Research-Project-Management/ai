"""Pure deterministic multi-file LaTeX source normalization and include-graph walker."""

from __future__ import annotations

import posixpath
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from ai.agents.editor.context.audit_models import IncludeIssue
from ai.agents.editor.context.latex_scanner import VERBATIM_ENVS, mask_environments
from ai.agents.editor.context.preamble import extract_preamble_summary

_DOC_CLASS_RE = re.compile(r"\\documentclass(?:\[[^\]]*\])?\{([^}]+)\}")
_BEGIN_DOC_RE = re.compile(r"\\begin\{document\}")
_REQUIRE_PKG_RE = re.compile(r"\\(?:usepackage|RequirePackage)(?:\[[^\]]*\])?\{([^}]+)\}")
_INCLUDE_RE = re.compile(r"\\(?:input|include|subfile)\{([^}]+)\}")
_SECTION_RE = re.compile(
    r"\\(part|chapter|section|subsection|subsubsection)\*?(?:\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})+)\}"
)
_NON_PROSE_MACRO_RE = re.compile(
    r"\\(?:cite[a-zA-Z]*|ref|eqref|pageref|autoref|nameref|cref|Cref|label|"
    r"begin|end|input|include|subfile|usepackage|RequirePackage|documentclass|"
    r"bibliographystyle|bibliography|addbibresource|captionof)\*?(?:\[[^\]]*\])?\{[^}]*\}"
)
_CMD_NAME_RE = re.compile(r"\\[A-Za-z@]+\*?")
_WORD_TOKEN_RE = re.compile(r"[A-Za-zÀ-ỹ0-9][A-Za-zÀ-ỹ0-9'-]*")


@dataclass
class OutlineDraft:
    """Mutable intermediate heading node accumulating word counts during walk."""

    file_path: str
    line: int
    level: str
    title: str
    words: int = 0


@dataclass
class WalkResult:
    """Complete reading-order, include-graph, and word-count walk output."""

    source_lines: Mapping[str, list[str]]
    root_file: str | None
    has_documentclass: bool
    reached: set[str] = field(default_factory=set)
    visited: set[str] = field(default_factory=set)
    reading_order_files: list[str] = field(default_factory=list)
    missing_includes: list[IncludeIssue] = field(default_factory=list)
    outline_drafts: list[OutlineDraft] = field(default_factory=list)
    preamble_packages_seen: dict[str, None] = field(default_factory=dict)
    total_words_approx: int = 0


def _normalize_path(raw: str) -> str:
    clean = raw.strip().replace("\\", "/").lstrip("/")
    while clean.startswith("./"):
        clean = clean[2:]
    return posixpath.normpath(clean) if clean else ""


def _resolve_include_target(
    parent_path: str,
    raw_target: str,
    known_paths: Mapping[str, list[str]],
) -> str | None:
    target = raw_target.strip().replace("\\", "/")
    while target.startswith("./"):
        target = target[2:]
    target = target.lstrip("/")
    if not target:
        return None

    parent_dir = parent_path.rsplit("/", 1)[0] if "/" in parent_path else ""
    candidates: list[str] = [target]
    if not target.lower().endswith((".tex", ".sty")):
        candidates.append(f"{target}.tex")
    if parent_dir:
        joined = posixpath.normpath(f"{parent_dir}/{target}")
        candidates.append(joined)
        if not joined.lower().endswith((".tex", ".sty")):
            candidates.append(f"{joined}.tex")

    for cand in candidates:
        norm = _normalize_path(cand)
        if norm in known_paths:
            return norm
    return None


def _count_latex_words(clean_line: str) -> int:
    text = _NON_PROSE_MACRO_RE.sub(" ", clean_line)
    text = _CMD_NAME_RE.sub(" ", text)
    return len(_WORD_TOKEN_RE.findall(text))


def _collect_packages(lines: list[str]) -> list[str]:
    packages: list[str] = []
    for line in lines:
        for match in _REQUIRE_PKG_RE.finditer(line):
            for pkg in match.group(1).split(","):
                clean = pkg.strip()
                if clean:
                    packages.append(clean)
    return packages


def prepare_sources(
    files: Mapping[str, str],
) -> tuple[dict[str, list[str]], set[str], dict[str, str]]:
    """Filter `.tex`/`.sty` sources and mask verbatim/comment environments per file."""
    source_lines: dict[str, list[str]] = {}
    tex_paths: set[str] = set()
    raw_contents: dict[str, str] = {}

    for raw_path, raw_text in files.items():
        norm_path = _normalize_path(raw_path)
        if not norm_path:
            continue
        lower = norm_path.lower()
        filename = norm_path.rsplit("/", 1)[-1]
        is_bib = lower.endswith(".bib")
        is_tex_or_sty = lower.endswith((".tex", ".sty")) or "." not in filename
        if is_bib or not is_tex_or_sty:
            continue
        source_lines[norm_path] = mask_environments(raw_text, skipped_envs=VERBATIM_ENVS)
        raw_contents[norm_path] = raw_text
        if not lower.endswith(".sty"):
            tex_paths.add(norm_path)

    return source_lines, tex_paths, raw_contents


def resolve_root_file(
    tex_paths: set[str],
    source_lines: Mapping[str, list[str]],
    root_file_hint: str | None,
) -> tuple[str | None, bool]:
    """Resolve the entry `.tex` file and whether it declares `\\documentclass`."""
    norm_hint = _normalize_path(root_file_hint) if root_file_hint else None
    docclass_files = sorted(
        p for p in tex_paths if any(_DOC_CLASS_RE.search(ln) for ln in source_lines[p])
    )

    if docclass_files:
        if norm_hint and norm_hint in docclass_files:
            return norm_hint, True
        if "main.tex" in docclass_files:
            return "main.tex", True
        return docclass_files[0], True

    ordered_tex = sorted(tex_paths)
    if norm_hint and norm_hint in tex_paths:
        return norm_hint, False
    if "main.tex" in tex_paths:
        return "main.tex", False
    if ordered_tex:
        return ordered_tex[0], False
    return None, False


def collect_preamble_packages(
    *,
    has_documentclass: bool,
    root_file: str | None,
    raw_contents: Mapping[str, str],
    source_lines: Mapping[str, list[str]],
) -> dict[str, None]:
    """Collect declared packages from the root file's preamble."""
    seen: dict[str, None] = {}
    if not has_documentclass or root_file is None:
        return seen
    root_summary = extract_preamble_summary(raw_contents[root_file])
    for pkg in root_summary.packages:
        seen.setdefault(pkg, None)
    begin_doc_idx = root_summary.begin_document_line or len(source_lines[root_file])
    for pkg in _collect_packages(source_lines[root_file][:begin_doc_idx]):
        seen.setdefault(pkg, None)
    return seen


def _walk_file(
    walk: WalkResult,
    path: str,
    *,
    track_reached: bool,
    collect_preamble_includes: bool,
) -> None:
    if path in walk.visited:
        return
    walk.visited.add(path)
    if track_reached:
        walk.reached.add(path)
    walk.reading_order_files.append(path)

    before_begin_doc = True
    for idx, line in enumerate(walk.source_lines[path], start=1):
        if _BEGIN_DOC_RE.search(line):
            before_begin_doc = False

        sec_m = _SECTION_RE.search(line)
        if sec_m:
            title_clean = re.sub(r"\s+", " ", sec_m.group(2)).strip()
            walk.outline_drafts.append(
                OutlineDraft(
                    file_path=path,
                    line=idx,
                    level=sec_m.group(1),
                    title=title_clean,
                )
            )
        else:
            words = _count_latex_words(line)
            if words > 0:
                walk.total_words_approx += words
                if walk.outline_drafts:
                    walk.outline_drafts[-1].words += words

        for inc_m in _INCLUDE_RE.finditer(line):
            raw_target = inc_m.group(1).strip()
            resolved = _resolve_include_target(path, raw_target, walk.source_lines)
            if resolved is None:
                walk.missing_includes.append(
                    IncludeIssue(file_path=path, line=idx, target=raw_target)
                )
                continue
            if (
                collect_preamble_includes
                and walk.has_documentclass
                and before_begin_doc
                and path == walk.root_file
            ):
                for pkg in _collect_packages(walk.source_lines[resolved]):
                    walk.preamble_packages_seen.setdefault(pkg, None)
            _walk_file(
                walk,
                resolved,
                track_reached=track_reached,
                collect_preamble_includes=False,
            )


def walk_reading_order(
    *,
    source_lines: Mapping[str, list[str]],
    root_file: str | None,
    has_documentclass: bool,
    preamble_packages_seen: dict[str, None],
) -> WalkResult:
    """Traverse entry includes and unreached sources in deterministic reading order."""
    walk = WalkResult(
        source_lines=source_lines,
        root_file=root_file,
        has_documentclass=has_documentclass,
        preamble_packages_seen=preamble_packages_seen,
    )
    if root_file is not None and root_file in source_lines:
        _walk_file(
            walk,
            root_file,
            track_reached=True,
            collect_preamble_includes=True,
        )
    for path in sorted(source_lines):
        if path not in walk.visited:
            _walk_file(
                walk,
                path,
                track_reached=False,
                collect_preamble_includes=False,
            )
    return walk


__all__ = [
    "OutlineDraft",
    "WalkResult",
    "collect_preamble_packages",
    "prepare_sources",
    "resolve_root_file",
    "walk_reading_order",
]
