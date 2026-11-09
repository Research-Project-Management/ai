"""Pure deterministic multi-file LaTeX page structure, x-ref & package auditor."""

from __future__ import annotations

import posixpath
import re
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from ai.agents.editor.context.latex_scanner import VERBATIM_ENVS, mask_environments
from ai.agents.editor.context.preamble import extract_preamble_summary

_DOC_CLASS_RE = re.compile(r"\\documentclass(?:\[[^\]]*\])?\{([^}]+)\}")
_BEGIN_DOC_RE = re.compile(r"\\begin\{document\}")
_REQUIRE_PKG_RE = re.compile(r"\\(?:usepackage|RequirePackage)(?:\[[^\]]*\])?\{([^}]+)\}")
_INCLUDE_RE = re.compile(r"\\(?:input|include|subfile)\{([^}]+)\}")
_SECTION_RE = re.compile(
    r"\\(part|chapter|section|subsection|subsubsection)\*?(?:\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})+)\}"
)
_LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
_REF_RE = re.compile(r"\\(ref|eqref|pageref|autoref|nameref|cref|Cref)\{([^}]+)\}")
_NON_PROSE_MACRO_RE = re.compile(
    r"\\(?:cite[a-zA-Z]*|ref|eqref|pageref|autoref|nameref|cref|Cref|label|"
    r"begin|end|input|include|subfile|usepackage|RequirePackage|documentclass|"
    r"bibliographystyle|bibliography|addbibresource|captionof)\*?(?:\[[^\]]*\])?\{[^}]*\}"
)
_CMD_NAME_RE = re.compile(r"\\[A-Za-z@]+\*?")
_WORD_TOKEN_RE = re.compile(r"[A-Za-zÀ-ỹ0-9][A-Za-zÀ-ỹ0-9'-]*")


@dataclass(frozen=True)
class _PackageRule:
    package: str
    satisfied_by: frozenset[str]
    pattern: re.Pattern[str]
    trigger_label: str


_PACKAGE_RULES: tuple[_PackageRule, ...] = (
    _PackageRule(
        package="booktabs",
        satisfied_by=frozenset({"booktabs"}),
        pattern=re.compile(r"\\(?:toprule|midrule|bottomrule)\b"),
        trigger_label="\\toprule/\\midrule/\\bottomrule",
    ),
    _PackageRule(
        package="graphicx",
        satisfied_by=frozenset({"graphicx"}),
        pattern=re.compile(r"\\includegraphics\b"),
        trigger_label="\\includegraphics",
    ),
    _PackageRule(
        package="hyperref",
        satisfied_by=frozenset({"hyperref"}),
        pattern=re.compile(r"\\(?:href|autoref)\b"),
        trigger_label="\\href/\\autoref",
    ),
    _PackageRule(
        package="hyperref",
        satisfied_by=frozenset({"hyperref", "url"}),
        pattern=re.compile(r"\\url\b"),
        trigger_label="\\url",
    ),
    _PackageRule(
        package="cleveref",
        satisfied_by=frozenset({"cleveref"}),
        pattern=re.compile(r"\\(?:cref|Cref)\b"),
        trigger_label="\\cref/\\Cref",
    ),
    _PackageRule(
        package="amsmath",
        satisfied_by=frozenset({"amsmath", "mathtools"}),
        pattern=re.compile(
            r"\\begin\{(?:align\*?|gather\*?|multline\*?)\}|\\eqref\b|\\operatorname\b"
        ),
        trigger_label="align/gather/\\eqref/\\operatorname",
    ),
    _PackageRule(
        package="amssymb",
        satisfied_by=frozenset({"amssymb", "amsfonts"}),
        pattern=re.compile(r"\\(?:mathbb|mathfrak)\b"),
        trigger_label="\\mathbb/\\mathfrak",
    ),
    _PackageRule(
        package="multirow",
        satisfied_by=frozenset({"multirow"}),
        pattern=re.compile(r"\\multirow\b"),
        trigger_label="\\multirow",
    ),
    _PackageRule(
        package="tabularx",
        satisfied_by=frozenset({"tabularx"}),
        pattern=re.compile(r"\\begin\{tabularx\}"),
        trigger_label="tabularx environment",
    ),
    _PackageRule(
        package="siunitx",
        satisfied_by=frozenset({"siunitx"}),
        pattern=re.compile(r"\\(?:SI|si|num|qty)\{"),
        trigger_label="\\SI/\\si/\\num/\\qty",
    ),
    _PackageRule(
        package="tikz",
        satisfied_by=frozenset({"tikz"}),
        pattern=re.compile(r"\\begin\{tikzpicture\}"),
        trigger_label="tikzpicture environment",
    ),
    _PackageRule(
        package="xcolor",
        satisfied_by=frozenset({"xcolor", "color"}),
        pattern=re.compile(r"\\(?:textcolor|colorbox)\b"),
        trigger_label="\\textcolor/\\colorbox",
    ),
    _PackageRule(
        package="subcaption",
        satisfied_by=frozenset({"subcaption", "subfig"}),
        pattern=re.compile(r"\\begin\{subfigure\}|\\subcaption\b"),
        trigger_label="subfigure/\\subcaption",
    ),
)


class OutlineSectionNode(BaseModel):
    """Single structural heading node across multi-file page reading order."""

    model_config = ConfigDict(frozen=True)

    file_path: str
    line: int
    level: str
    title: str
    approx_words: int = 0


class DuplicateLabelIssue(BaseModel):
    """A `\\label{...}` defined more than once across the page."""

    model_config = ConfigDict(frozen=True)

    label: str
    locations: tuple[str, ...]


class ReferenceIssue(BaseModel):
    """A cross-reference command pointing to an undefined `\\label{...}`."""

    model_config = ConfigDict(frozen=True)

    file_path: str
    line: int
    command: str
    key: str


class IncludeIssue(BaseModel):
    """An unresolved `\\input` / `\\include` / `\\subfile` target."""

    model_config = ConfigDict(frozen=True)

    file_path: str
    line: int
    target: str


class MissingPackageIssue(BaseModel):
    """A macro or environment used without declaring its required package in preamble."""

    model_config = ConfigDict(frozen=True)

    package: str
    trigger: str
    file_path: str
    line: int


class PageStructureReport(BaseModel):
    """Full structural, x-ref, include-graph & package report (0 LLM / 0 I/O)."""

    model_config = ConfigDict(frozen=True)

    root_file: str | None = None
    total_tex_files: int = 0
    total_words_approx: int = 0
    outline: tuple[OutlineSectionNode, ...] = ()
    total_labels: int = 0
    total_refs: int = 0
    duplicate_labels: tuple[DuplicateLabelIssue, ...] = ()
    undefined_refs: tuple[ReferenceIssue, ...] = ()
    unreferenced_labels: tuple[str, ...] = ()
    missing_includes: tuple[IncludeIssue, ...] = ()
    orphan_tex_files: tuple[str, ...] = ()
    loaded_packages: tuple[str, ...] = ()
    missing_packages: tuple[MissingPackageIssue, ...] = ()


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


@dataclass
class _OutlineDraft:
    file_path: str
    line: int
    level: str
    title: str
    words: int = 0


@dataclass
class _ScanWalkState:
    source_lines: Mapping[str, list[str]]
    root_file: str | None
    has_documentclass: bool
    reached: set[str]
    visited: set[str]
    reading_order_files: list[str]
    missing_includes: list[IncludeIssue]
    outline_drafts: list[_OutlineDraft]
    preamble_packages_seen: dict[str, None]
    total_words_approx: int = 0


def _prepare_page_sources(
    files: Mapping[str, str],
) -> tuple[dict[str, list[str]], set[str], dict[str, str]]:
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


def _resolve_root_file(
    tex_paths: set[str],
    source_lines: Mapping[str, list[str]],
    root_file_hint: str | None,
) -> tuple[str | None, bool]:
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


def _collect_root_preamble_packages(
    *,
    has_documentclass: bool,
    root_file: str | None,
    raw_contents: Mapping[str, str],
    source_lines: Mapping[str, list[str]],
) -> dict[str, None]:
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
    walk: _ScanWalkState,
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
                _OutlineDraft(
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


def _scan_reading_order(
    *,
    source_lines: Mapping[str, list[str]],
    root_file: str | None,
    has_documentclass: bool,
    preamble_packages_seen: dict[str, None],
) -> _ScanWalkState:
    walk = _ScanWalkState(
        source_lines=source_lines,
        root_file=root_file,
        has_documentclass=has_documentclass,
        reached=set(),
        visited=set(),
        reading_order_files=[],
        missing_includes=[],
        outline_drafts=[],
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


def _audit_cross_references(
    reading_order_files: list[str],
    source_lines: Mapping[str, list[str]],
    *,
    include_unreferenced_labels: bool,
    max_issues_per_category: int,
) -> tuple[
    int,
    int,
    tuple[DuplicateLabelIssue, ...],
    tuple[ReferenceIssue, ...],
    tuple[str, ...],
]:
    label_locations: dict[str, list[str]] = {}
    ref_occurrences: list[ReferenceIssue] = []
    referenced_keys: set[str] = set()

    for path in reading_order_files:
        for idx, line in enumerate(source_lines[path], start=1):
            for lbl_m in _LABEL_RE.finditer(line):
                key = lbl_m.group(1).strip()
                if key:
                    label_locations.setdefault(key, []).append(f"{path}:L{idx}")

            for ref_m in _REF_RE.finditer(line):
                cmd = f"\\{ref_m.group(1)}"
                for part in ref_m.group(2).split(","):
                    ref_key = part.strip()
                    if ref_key:
                        referenced_keys.add(ref_key)
                        ref_occurrences.append(
                            ReferenceIssue(
                                file_path=path,
                                line=idx,
                                command=cmd,
                                key=ref_key,
                            )
                        )

    duplicate_labels = tuple(
        DuplicateLabelIssue(label=lbl, locations=tuple(locs))
        for lbl, locs in label_locations.items()
        if len(locs) >= 2
    )[:max_issues_per_category]

    defined_keys = set(label_locations)
    undefined_refs = tuple(
        ref for ref in ref_occurrences if ref.key not in defined_keys
    )[:max_issues_per_category]

    unreferenced_labels: tuple[str, ...] = ()
    if include_unreferenced_labels:
        unreferenced_labels = tuple(
            sorted(k for k in defined_keys if k not in referenced_keys)
        )[:max_issues_per_category]

    return (
        len(label_locations),
        len(ref_occurrences),
        duplicate_labels,
        undefined_refs,
        unreferenced_labels,
    )


def _audit_missing_packages(
    reading_order_files: list[str],
    source_lines: Mapping[str, list[str]],
    preamble_packages_seen: Mapping[str, None],
    *,
    max_issues_per_category: int,
) -> tuple[MissingPackageIssue, ...]:
    loaded_set_lower = {p.lower() for p in preamble_packages_seen}
    missing_packages: list[MissingPackageIssue] = []
    seen_missing_pkgs: set[str] = set()

    for path in reading_order_files:
        if not path.lower().endswith(".tex") and "." in path.rsplit("/", 1)[-1]:
            continue
        for idx, line in enumerate(source_lines[path], start=1):
            for rule in _PACKAGE_RULES:
                if rule.package in seen_missing_pkgs:
                    continue
                if any(pkg in loaded_set_lower for pkg in rule.satisfied_by):
                    continue
                if rule.pattern.search(line):
                    seen_missing_pkgs.add(rule.package)
                    missing_packages.append(
                        MissingPackageIssue(
                            package=rule.package,
                            trigger=rule.trigger_label,
                            file_path=path,
                            line=idx,
                        )
                    )
                    if len(missing_packages) >= max_issues_per_category:
                        return tuple(missing_packages)
    return tuple(missing_packages)


def audit_page_structure(
    files: Mapping[str, str],
    *,
    root_file_hint: str | None = None,
    include_unreferenced_labels: bool = False,
    max_issues_per_category: int = 25,
) -> PageStructureReport:
    """Audit multi-file LaTeX structure, outline, x-refs, includes, and packages."""
    source_lines, tex_paths, raw_contents = _prepare_page_sources(files)
    if not source_lines:
        return PageStructureReport()

    root_file, has_documentclass = _resolve_root_file(tex_paths, source_lines, root_file_hint)
    preamble_packages_seen = _collect_root_preamble_packages(
        has_documentclass=has_documentclass,
        root_file=root_file,
        raw_contents=raw_contents,
        source_lines=source_lines,
    )
    walk = _scan_reading_order(
        source_lines=source_lines,
        root_file=root_file,
        has_documentclass=has_documentclass,
        preamble_packages_seen=preamble_packages_seen,
    )

    orphan_tex_files: tuple[str, ...] = ()
    if has_documentclass and len(tex_paths) > 1:
        orphan_tex_files = tuple(sorted(p for p in tex_paths if p not in walk.reached))[
            :max_issues_per_category
        ]

    total_labels, total_refs, duplicate_labels, undefined_refs, unreferenced_labels = (
        _audit_cross_references(
            walk.reading_order_files,
            source_lines,
            include_unreferenced_labels=include_unreferenced_labels,
            max_issues_per_category=max_issues_per_category,
        )
    )

    missing_packages = (
        _audit_missing_packages(
            walk.reading_order_files,
            source_lines,
            walk.preamble_packages_seen,
            max_issues_per_category=max_issues_per_category,
        )
        if has_documentclass
        else ()
    )

    outline = tuple(
        OutlineSectionNode(
            file_path=item.file_path,
            line=item.line,
            level=item.level,
            title=item.title,
            approx_words=item.words,
        )
        for item in walk.outline_drafts
    )

    return PageStructureReport(
        root_file=root_file,
        total_tex_files=len(tex_paths),
        total_words_approx=walk.total_words_approx,
        outline=outline,
        total_labels=total_labels,
        total_refs=total_refs,
        duplicate_labels=duplicate_labels,
        undefined_refs=undefined_refs,
        unreferenced_labels=unreferenced_labels,
        missing_includes=tuple(walk.missing_includes[:max_issues_per_category]),
        orphan_tex_files=orphan_tex_files,
        loaded_packages=tuple(walk.preamble_packages_seen.keys()),
        missing_packages=missing_packages,
    )
