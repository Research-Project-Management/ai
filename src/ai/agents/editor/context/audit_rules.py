"""Strategy rule pipeline for LaTeX structure, cross-reference, and package checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from ai.agents.editor.context.audit_models import (
    AuditAccumulator,
    AuditContext,
    AuditOptions,
    DuplicateLabelIssue,
    MissingPackageIssue,
    ReferenceIssue,
)

_LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
_REF_RE = re.compile(r"\\(ref|eqref|pageref|autoref|nameref|cref|Cref)\{([^}]+)\}")


@dataclass(frozen=True)
class _PackageSpec:
    package: str
    satisfied_by: frozenset[str]
    pattern: re.Pattern[str]
    trigger_label: str


_PACKAGE_SPECS: tuple[_PackageSpec, ...] = (
    _PackageSpec(
        package="booktabs",
        satisfied_by=frozenset({"booktabs"}),
        pattern=re.compile(r"\\(?:toprule|midrule|bottomrule)\b"),
        trigger_label="\\toprule/\\midrule/\\bottomrule",
    ),
    _PackageSpec(
        package="graphicx",
        satisfied_by=frozenset({"graphicx"}),
        pattern=re.compile(r"\\includegraphics\b"),
        trigger_label="\\includegraphics",
    ),
    _PackageSpec(
        package="hyperref",
        satisfied_by=frozenset({"hyperref"}),
        pattern=re.compile(r"\\(?:href|autoref)\b"),
        trigger_label="\\href/\\autoref",
    ),
    _PackageSpec(
        package="hyperref",
        satisfied_by=frozenset({"hyperref", "url"}),
        pattern=re.compile(r"\\url\b"),
        trigger_label="\\url",
    ),
    _PackageSpec(
        package="cleveref",
        satisfied_by=frozenset({"cleveref"}),
        pattern=re.compile(r"\\(?:cref|Cref)\b"),
        trigger_label="\\cref/\\Cref",
    ),
    _PackageSpec(
        package="amsmath",
        satisfied_by=frozenset({"amsmath", "mathtools"}),
        pattern=re.compile(
            r"\\begin\{(?:align\*?|gather\*?|multline\*?)\}|\\eqref\b|\\operatorname\b"
        ),
        trigger_label="align/gather/\\eqref/\\operatorname",
    ),
    _PackageSpec(
        package="amssymb",
        satisfied_by=frozenset({"amssymb", "amsfonts"}),
        pattern=re.compile(r"\\(?:mathbb|mathfrak)\b"),
        trigger_label="\\mathbb/\\mathfrak",
    ),
    _PackageSpec(
        package="multirow",
        satisfied_by=frozenset({"multirow"}),
        pattern=re.compile(r"\\multirow\b"),
        trigger_label="\\multirow",
    ),
    _PackageSpec(
        package="tabularx",
        satisfied_by=frozenset({"tabularx"}),
        pattern=re.compile(r"\\begin\{tabularx\}"),
        trigger_label="tabularx environment",
    ),
    _PackageSpec(
        package="siunitx",
        satisfied_by=frozenset({"siunitx"}),
        pattern=re.compile(r"\\(?:SI|si|num|qty)\{"),
        trigger_label="\\SI/\\si/\\num/\\qty",
    ),
    _PackageSpec(
        package="tikz",
        satisfied_by=frozenset({"tikz"}),
        pattern=re.compile(r"\\begin\{tikzpicture\}"),
        trigger_label="tikzpicture environment",
    ),
    _PackageSpec(
        package="xcolor",
        satisfied_by=frozenset({"xcolor", "color"}),
        pattern=re.compile(r"\\(?:textcolor|colorbox)\b"),
        trigger_label="\\textcolor/\\colorbox",
    ),
    _PackageSpec(
        package="subcaption",
        satisfied_by=frozenset({"subcaption", "subfig"}),
        pattern=re.compile(r"\\begin\{subfigure\}|\\subcaption\b"),
        trigger_label="subfigure/\\subcaption",
    ),
)


class AuditRule(Protocol):
    """Independent structure audit strategy recording findings into `AuditAccumulator`."""

    def run(self, ctx: AuditContext, opts: AuditOptions, acc: AuditAccumulator) -> None: ...


class OrphanFileRule:
    """Identifies `.tex` files unreached from the document root include tree."""

    def run(self, ctx: AuditContext, opts: AuditOptions, acc: AuditAccumulator) -> None:
        if not (ctx.has_documentclass and len(ctx.tex_paths) > 1):
            return
        acc.orphan_tex_files = tuple(sorted(p for p in ctx.tex_paths if p not in ctx.reached))[
            : opts.max_issues_per_category
        ]


class CrossRefRule:
    """Audits duplicate `\\label{...}`, undefined references, and unreferenced labels."""

    def run(self, ctx: AuditContext, opts: AuditOptions, acc: AuditAccumulator) -> None:
        label_locations: dict[str, list[str]] = {}
        ref_occurrences: list[ReferenceIssue] = []
        referenced_keys: set[str] = set()

        for path in ctx.reading_order_files:
            for idx, line in enumerate(ctx.source_lines[path], start=1):
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

        defined_keys = set(label_locations)
        acc.total_labels = len(label_locations)
        acc.total_refs = len(ref_occurrences)
        acc.duplicate_labels = tuple(
            DuplicateLabelIssue(label=lbl, locations=tuple(locs))
            for lbl, locs in label_locations.items()
            if len(locs) >= 2
        )[: opts.max_issues_per_category]
        acc.undefined_refs = tuple(ref for ref in ref_occurrences if ref.key not in defined_keys)[
            : opts.max_issues_per_category
        ]
        if opts.include_unreferenced_labels:
            acc.unreferenced_labels = tuple(
                sorted(k for k in defined_keys if k not in referenced_keys)
            )[: opts.max_issues_per_category]


class PackageRule:
    """Detects LaTeX commands/environments missing required `\\usepackage` declarations."""

    def run(self, ctx: AuditContext, opts: AuditOptions, acc: AuditAccumulator) -> None:
        if not ctx.has_documentclass:
            return
        loaded_lower = {p.lower() for p in ctx.preamble_packages_seen}
        missing: list[MissingPackageIssue] = []
        seen_pkgs: set[str] = set()

        for path in ctx.reading_order_files:
            if not path.lower().endswith(".tex") and "." in path.rsplit("/", 1)[-1]:
                continue
            for idx, line in enumerate(ctx.source_lines[path], start=1):
                for spec in _PACKAGE_SPECS:
                    if spec.package in seen_pkgs:
                        continue
                    if any(pkg in loaded_lower for pkg in spec.satisfied_by):
                        continue
                    if spec.pattern.search(line):
                        seen_pkgs.add(spec.package)
                        missing.append(
                            MissingPackageIssue(
                                package=spec.package,
                                trigger=spec.trigger_label,
                                file_path=path,
                                line=idx,
                            )
                        )
                        if len(missing) >= opts.max_issues_per_category:
                            acc.missing_packages = tuple(missing)
                            return
        acc.missing_packages = tuple(missing)


DEFAULT_AUDIT_RULES: tuple[AuditRule, ...] = (
    OrphanFileRule(),
    CrossRefRule(),
    PackageRule(),
)

__all__ = [
    "DEFAULT_AUDIT_RULES",
    "AuditRule",
    "CrossRefRule",
    "OrphanFileRule",
    "PackageRule",
]
