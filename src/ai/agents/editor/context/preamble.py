"""Pure deterministic LaTeX preamble and local symbol extraction helpers."""

from __future__ import annotations

import re

from ai.agents.editor.context.contracts import PreambleSummary
from ai.agents.editor.context.latex_scanner import (
    extract_cite_keys,
    extract_label_defs,
    normalize_newlines,
    strip_latex_comments,
)

_DOC_CLASS_RE = re.compile(r"\\documentclass(?:\[[^\]]*\])?\{([^}]+)\}")
_USEPACKAGE_RE = re.compile(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}")
_MACRO_RE = re.compile(
    r"\\(?:re|provide)?newcommand\*?(?:\[[^\]]*\])?\{?(\\[A-Za-z@]+)\}?"
    r"|\\DeclareMathOperator\*?\{?(\\[A-Za-z@]+)\}?"
)
_BEGIN_DOC_RE = re.compile(r"\\begin\{document\}")


def extract_preamble_summary(content: str, *, max_excerpt_chars: int = 1500) -> PreambleSummary:
    """Extract documentclass, packages, custom macros, and compact preamble excerpt."""
    if not content:
        return PreambleSummary()

    normalized = normalize_newlines(content)
    stripped_lines = strip_latex_comments(normalized).splitlines()
    raw_lines = normalized.splitlines()

    begin_doc_line: int | None = None
    for idx, clean_line in enumerate(stripped_lines, start=1):
        if _BEGIN_DOC_RE.search(clean_line):
            begin_doc_line = idx
            break

    cutoff = begin_doc_line - 1 if begin_doc_line is not None else min(len(raw_lines), 80)
    preamble_raw = "\n".join(raw_lines[:cutoff])
    preamble_clean = "\n".join(stripped_lines[:cutoff])

    doc_class_match = _DOC_CLASS_RE.search(preamble_clean)
    document_class = doc_class_match.group(1).strip() if doc_class_match else ""

    packages_seen: dict[str, None] = {}
    for match in _USEPACKAGE_RE.finditer(preamble_clean):
        for pkg in match.group(1).split(","):
            clean_pkg = pkg.strip()
            if clean_pkg:
                packages_seen.setdefault(clean_pkg, None)

    macros_seen: dict[str, None] = {}
    for match in _MACRO_RE.finditer(preamble_clean):
        name = (match.group(1) or match.group(2) or "").strip()
        if name:
            macros_seen.setdefault(name, None)

    excerpt = preamble_raw.strip()
    if len(excerpt) > max_excerpt_chars:
        excerpt = excerpt[:max_excerpt_chars].rstrip() + "\n% ... (preamble truncated)"

    return PreambleSummary(
        document_class=document_class,
        begin_document_line=begin_doc_line,
        packages=tuple(packages_seen.keys()),
        custom_macros=tuple(list(macros_seen.keys())[:40]),
        raw_excerpt=excerpt if begin_doc_line is not None or document_class or packages_seen else "",
    )


def extract_local_symbols(content: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Extract unique `\\label{...}` and `\\cite*{...}` keys present in a LaTeX source."""
    if not content:
        return (), ()
    clean = strip_latex_comments(content)
    return extract_label_defs(clean), extract_cite_keys(clean)
