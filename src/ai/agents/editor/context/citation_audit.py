"""Deterministic citation reviewer detecting missing \\cite{} claims and .bib mismatches."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from ai.agents.editor.context.latex_scanner import (
    extract_cite_keys,
    mask_environments,
    strip_latex_comments,
)

_BEGIN_DOC_RE = re.compile(r"\\begin\{document\}")
_BEGIN_ENV_RE = re.compile(r"\\begin\{([^{}\s]+)\}")
_END_ENV_RE = re.compile(r"\\end\{([^{}\s]+)\}")
_SECTION_RE = re.compile(r"\\(?:part|chapter|section|subsection|subsubsection)\*?\{([^}]*)\}")
_ANY_CITE_OR_FOOTNOTE_RE = re.compile(
    r"\\(?:[a-zA-Z]*cite[a-zA-Z]*|footnote)\*?(?:\[[^\]]*\])*\{[^}]+\}"
)

_SKIPPED_ENVIRONMENTS: frozenset[str] = frozenset(
    {
        "abstract",
        "verbatim",
        "lstlisting",
        "minted",
        "comment",
        "equation",
        "equation*",
        "align",
        "align*",
        "gather",
        "gather*",
        "tabular",
        "tabularx",
        "table",
        "table*",
        "figure",
        "figure*",
        "tikzpicture",
        "thebibliography",
    }
)

_SKIPPED_SECTIONS: tuple[str, ...] = ("abstract", "acknowledg", "appendix")

_LITERATURE_CUE_RE = re.compile(
    r"\b(?:"
    r"prior work|previous work|previous studies|earlier work|recent work|recent studies|"
    r"the literature|as shown by|has been shown|have shown|it is well known|"
    r"widely adopted|widely used|state-of-the-art|classic approach|existing methods|"
    r"previous approaches"
    r")\b"
    r"|et\s+al\."
    r"|các nghiên cứu trước|nghiên cứu trước đây|theo tài liệu|trạng thái tiên tiến",
    re.IGNORECASE,
)

_COMPARATIVE_CUE_RE = re.compile(
    r"\b(?:outperform\w*|surpass\w*|benchmark\w*|baseline\w*|speedup)\b"
    r"|\b(?:achieves|improves\s+by|reduces\s+[^.]{1,40}\s+by)\b"
    r"|\b\d+(?:\.\d+)?\s*%",
    re.IGNORECASE,
)

_SELF_REFERENTIAL_RE = re.compile(
    r"\b(?:our method|our approach|our model|we propose|in this paper|in section|in table|"
    r"table~?\\ref|figure~?\\ref|fig\.?~?\\ref|eq\.?~?\\eqref)\b"
    r"|phương pháp của chúng tôi|chúng tôi đề xuất|trong bài báo này|bảng~?\\ref|hình~?\\ref",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<!\be\.g)(?<!\bi\.e)(?<!\bet al)(?<!\bvs)(?<!\bFig)(?<!\bEq)(?<!\bSec)(?<!\bNo)"
    r"(?<=[.!?])\s+(?=[A-ZÀ-Ỹ0-9\\])"
)


class CitationGapCandidate(BaseModel):
    """A concrete page sentence likely needing a \\cite{...} key."""

    model_config = ConfigDict(frozen=True)

    line: int
    section: str = ""
    sentence: str
    reason: str


class CitationAuditReport(BaseModel):
    """Summary report produced by `audit_citation_gaps`."""

    model_config = ConfigDict(frozen=True)

    file_path: str
    total_citations_used: int
    total_bib_entries: int
    undefined_keys: tuple[str, ...] = ()
    unused_bib_keys: tuple[str, ...] = ()
    unsupported_claims: tuple[CitationGapCandidate, ...] = ()


def _is_structural_directive(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return stripped.startswith(
        (
            "\\documentclass",
            "\\usepackage",
            "\\begin{",
            "\\end{",
            "\\title",
            "\\author",
            "\\date",
            "\\maketitle",
            "\\bibliography",
            "\\bibliographystyle",
            "\\addbibresource",
            "\\input",
            "\\include",
            "\\label",
            "\\caption",
            "\\section",
            "\\subsection",
            "\\subsubsection",
            "\\chapter",
            "\\part",
            "\\[",
            "\\]",
            "$$",
        )
    )


def _compare_bib_keys(
    used_keys_ordered: tuple[str, ...],
    known_bib_keys: frozenset[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not known_bib_keys:
        return (), ()
    known_lower = {k.lower() for k in known_bib_keys}
    used_lower = {k.lower() for k in used_keys_ordered}
    undefined_keys = tuple(sorted(k for k in used_keys_ordered if k.lower() not in known_lower))
    unused_bib_keys = tuple(sorted(k for k in known_bib_keys if k.lower() not in used_lower)[:40])
    return undefined_keys, unused_bib_keys


def _extract_body_paragraphs(lines: list[str]) -> list[tuple[int, str, str]]:
    start_idx = 0
    for idx, line in enumerate(lines):
        if _BEGIN_DOC_RE.search(line):
            start_idx = idx + 1
            break

    paragraphs: list[tuple[int, str, str]] = []
    current_section = ""
    para_lines: list[str] = []
    para_start_line = 0

    for idx in range(start_idx, len(lines)):
        line_num = idx + 1
        line = lines[idx]
        stripped = line.strip()

        sec_m = _SECTION_RE.search(line)
        is_break = (
            sec_m is not None
            or _BEGIN_ENV_RE.search(line) is not None
            or _END_ENV_RE.search(line) is not None
            or not stripped
            or _is_structural_directive(stripped)
        )
        if is_break:
            if para_lines:
                joined = " ".join(ln.strip() for ln in para_lines if ln.strip())
                if joined:
                    paragraphs.append((para_start_line, current_section, joined))
                para_lines = []
                para_start_line = 0
            if sec_m is not None:
                current_section = sec_m.group(1).strip()
            continue

        if not para_lines:
            para_start_line = line_num
        para_lines.append(stripped)

    if para_lines:
        joined = " ".join(ln.strip() for ln in para_lines if ln.strip())
        if joined:
            paragraphs.append((para_start_line, current_section, joined))

    return paragraphs


def _classify_uncited_sentence(sentence: str) -> str:
    if len(sentence) < 25 or _ANY_CITE_OR_FOOTNOTE_RE.search(sentence):
        return ""
    if _LITERATURE_CUE_RE.search(sentence):
        return "References prior literature without a \\cite{...}."
    if _COMPARATIVE_CUE_RE.search(sentence) and not _SELF_REFERENTIAL_RE.search(sentence):
        return "Empirical or comparative claim without a \\cite{...}."
    return ""


def _find_unsupported_claims(
    paragraphs: list[tuple[int, str, str]],
    *,
    max_candidates: int,
) -> tuple[CitationGapCandidate, ...]:
    candidates: list[CitationGapCandidate] = []
    for para_line, section, paragraph in paragraphs:
        if any(skip in section.lower() for skip in _SKIPPED_SECTIONS):
            continue
        for sent in _SENTENCE_SPLIT_RE.split(paragraph):
            clean_sent = sent.strip()
            reason = _classify_uncited_sentence(clean_sent)
            if not reason:
                continue
            candidates.append(
                CitationGapCandidate(
                    line=para_line,
                    section=section,
                    sentence=clean_sent[:280],
                    reason=reason,
                )
            )
            if len(candidates) >= max_candidates:
                return tuple(candidates)
    return tuple(candidates)


def audit_citation_gaps(
    content: str,
    *,
    file_path: str,
    known_bib_keys: frozenset[str] = frozenset(),
    max_candidates: int = 15,
) -> CitationAuditReport:
    """Scan a LaTeX file for undefined \\cite keys, unused .bib keys, and uncited claims."""
    no_comments = strip_latex_comments(content)
    used_keys_ordered = extract_cite_keys(no_comments)
    undefined_keys, unused_bib_keys = _compare_bib_keys(used_keys_ordered, known_bib_keys)
    lines = mask_environments(content, skipped_envs=_SKIPPED_ENVIRONMENTS)
    paragraphs = _extract_body_paragraphs(lines)
    candidates = _find_unsupported_claims(paragraphs, max_candidates=max_candidates)

    return CitationAuditReport(
        file_path=file_path,
        total_citations_used=len(used_keys_ordered),
        total_bib_entries=len(known_bib_keys),
        undefined_keys=undefined_keys,
        unused_bib_keys=unused_bib_keys,
        unsupported_claims=candidates,
    )
