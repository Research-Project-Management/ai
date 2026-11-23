"""Contracts and rule-pipeline state for multi-file LaTeX page structure auditing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict


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


@dataclass(frozen=True)
class AuditOptions:
    """Configuration options controlling structure audit rule outputs."""

    include_unreferenced_labels: bool = False
    max_issues_per_category: int = 25


@dataclass(frozen=True)
class AuditContext:
    """Read-only prepared workspace snapshot consumed by independent audit rules."""

    source_lines: Mapping[str, list[str]]
    tex_paths: set[str]
    root_file: str | None
    has_documentclass: bool
    reading_order_files: tuple[str, ...]
    reached: frozenset[str]
    preamble_packages_seen: Mapping[str, None]


@dataclass
class AuditAccumulator:
    """Mutable findings collector populated sequentially by `AuditRule` strategies."""

    total_labels: int = 0
    total_refs: int = 0
    duplicate_labels: tuple[DuplicateLabelIssue, ...] = ()
    undefined_refs: tuple[ReferenceIssue, ...] = ()
    unreferenced_labels: tuple[str, ...] = ()
    orphan_tex_files: tuple[str, ...] = ()
    missing_packages: tuple[MissingPackageIssue, ...] = ()
    extra_notes: list[str] = field(default_factory=list)


__all__ = [
    "AuditAccumulator",
    "AuditContext",
    "AuditOptions",
    "DuplicateLabelIssue",
    "IncludeIssue",
    "MissingPackageIssue",
    "OutlineSectionNode",
    "PageStructureReport",
    "ReferenceIssue",
]
