"""Facade orchestrating multi-file LaTeX page structure auditing via `AuditRule` strategies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ai.agents.editor.context.audit_models import (
    AuditAccumulator,
    AuditContext,
    AuditOptions,
    DuplicateLabelIssue,
    IncludeIssue,
    MissingPackageIssue,
    OutlineSectionNode,
    PageStructureReport,
    ReferenceIssue,
)
from ai.agents.editor.context.audit_rules import (
    DEFAULT_AUDIT_RULES,
    AuditRule,
)
from ai.agents.editor.context.audit_walker import (
    collect_preamble_packages,
    prepare_sources,
    resolve_root_file,
    walk_reading_order,
)


class StructureAuditor:
    """Facade running an extensible sequence of `AuditRule` strategies over walked sources."""

    def __init__(self, rules: Sequence[AuditRule] = DEFAULT_AUDIT_RULES) -> None:
        self._rules = tuple(rules)

    def audit(
        self,
        files: Mapping[str, str],
        *,
        root_file_hint: str | None = None,
        include_unreferenced_labels: bool = False,
        max_issues_per_category: int = 25,
    ) -> PageStructureReport:
        """Audit multi-file LaTeX structure, outline, x-refs, includes, and packages."""
        source_lines, tex_paths, raw_contents = prepare_sources(files)
        if not source_lines:
            return PageStructureReport()

        root_file, has_documentclass = resolve_root_file(tex_paths, source_lines, root_file_hint)
        preamble_packages_seen = collect_preamble_packages(
            has_documentclass=has_documentclass,
            root_file=root_file,
            raw_contents=raw_contents,
            source_lines=source_lines,
        )
        walk = walk_reading_order(
            source_lines=source_lines,
            root_file=root_file,
            has_documentclass=has_documentclass,
            preamble_packages_seen=preamble_packages_seen,
        )

        ctx = AuditContext(
            source_lines=source_lines,
            tex_paths=tex_paths,
            root_file=root_file,
            has_documentclass=has_documentclass,
            reading_order_files=tuple(walk.reading_order_files),
            reached=frozenset(walk.reached),
            preamble_packages_seen=walk.preamble_packages_seen,
        )
        opts = AuditOptions(
            include_unreferenced_labels=include_unreferenced_labels,
            max_issues_per_category=max_issues_per_category,
        )
        acc = AuditAccumulator()
        for rule in self._rules:
            rule.run(ctx, opts, acc)

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
            total_labels=acc.total_labels,
            total_refs=acc.total_refs,
            duplicate_labels=acc.duplicate_labels,
            undefined_refs=acc.undefined_refs,
            unreferenced_labels=acc.unreferenced_labels,
            missing_includes=tuple(walk.missing_includes[:max_issues_per_category]),
            orphan_tex_files=acc.orphan_tex_files,
            loaded_packages=tuple(walk.preamble_packages_seen.keys()),
            missing_packages=acc.missing_packages,
        )


_DEFAULT_AUDITOR = StructureAuditor()


def audit_page_structure(
    files: Mapping[str, str],
    *,
    root_file_hint: str | None = None,
    include_unreferenced_labels: bool = False,
    max_issues_per_category: int = 25,
) -> PageStructureReport:
    """Audit multi-file LaTeX structure, outline, x-refs, includes, and packages."""
    return _DEFAULT_AUDITOR.audit(
        files,
        root_file_hint=root_file_hint,
        include_unreferenced_labels=include_unreferenced_labels,
        max_issues_per_category=max_issues_per_category,
    )


__all__ = [
    "DuplicateLabelIssue",
    "IncludeIssue",
    "MissingPackageIssue",
    "OutlineSectionNode",
    "PageStructureReport",
    "ReferenceIssue",
    "StructureAuditor",
    "audit_page_structure",
]
