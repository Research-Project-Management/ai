"""Composable Specification/Rule pipeline for deterministic LaTeX patch validation (0 LLM)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ai.agents.editor.context.latex_scanner import (
    VERBATIM_ENVS,
    extract_cite_keys,
    extract_label_defs,
    extract_ref_uses,
    mask_environments,
)
from ai.agents.editor.patching.contracts import PatchValidationResult

_ENV_TOKEN_RE = re.compile(r"\\(begin|end)\{([^{}\s]+)\}")
_DOC_CLASS_RE = re.compile(r"\\documentclass(?:\[[^\]]*\])?\{[^}]+\}")
_USEPACKAGE_RE = re.compile(r"\\usepackage(?:\[[^\]]*\])?\{[^}]+\}")


@dataclass(frozen=True)
class ValidationContext:
    """Normalized pre/post patch content and project symbol sets passed to validation rules."""

    orig_clean: str
    patched_clean: str
    known_cite_keys: frozenset[str] = frozenset()
    known_labels: frozenset[str] = frozenset()


class LatexValidationRule(Protocol):
    """Rule contract returning `(errors, warnings)` pairs for a candidate patch."""

    def check(self, ctx: ValidationContext) -> tuple[tuple[str, ...], tuple[str, ...]]: ...


def _count_env_mismatches(clean_tex: str) -> tuple[int, list[str]]:
    stack: list[str] = []
    mismatches = 0
    notes: list[str] = []

    for match in _ENV_TOKEN_RE.finditer(clean_tex):
        kind, env_name = match.group(1), match.group(2).strip()
        if kind == "begin":
            stack.append(env_name)
        elif stack and stack[-1] == env_name:
            stack.pop()
        elif env_name in stack:
            while stack and stack[-1] != env_name:
                unclosed = stack.pop()
                mismatches += 1
                notes.append(f"unclosed `\\begin{{{unclosed}}}` before `\\end{{{env_name}}}`")
            if stack:
                stack.pop()
        else:
            mismatches += 1
            notes.append(f"stray `\\end{{{env_name}}}` with no matching `\\begin{{{env_name}}}`")

    for leftover in stack:
        mismatches += 1
        notes.append(f"unclosed `\\begin{{{leftover}}}`")

    return mismatches, notes


def _scan_delimiter_deltas(clean_tex: str) -> tuple[int, int]:
    """Return `(brace_delta, display_math_bracket_delta)` respecting backslash parity."""
    brace_delta = 0
    bracket_delta = 0
    backslashes = 0

    for ch in clean_tex:
        if ch == "\\":
            backslashes += 1
            continue

        is_escaped = backslashes % 2 == 1
        backslashes = 0

        if ch == "{" and not is_escaped:
            brace_delta += 1
        elif ch == "}" and not is_escaped:
            brace_delta -= 1
        elif ch == "[" and is_escaped:
            bracket_delta += 1
        elif ch == "]" and is_escaped:
            bracket_delta -= 1

    return brace_delta, bracket_delta


class EnvironmentBalanceRule:
    """Reject patches introducing net new `\\begin{env}` / `\\end{env}` mismatches."""

    def check(self, ctx: ValidationContext) -> tuple[tuple[str, ...], tuple[str, ...]]:
        orig_mismatches, _ = _count_env_mismatches(ctx.orig_clean)
        patched_mismatches, notes = _count_env_mismatches(ctx.patched_clean)
        if patched_mismatches > orig_mismatches:
            detail = "; ".join(notes[:3]) or "environment mismatch"
            return (f"Unbalanced LaTeX environment introduced ({detail}).",), ()
        return (), ()


class BraceBalanceRule:
    """Reject patches introducing net new unbalanced `{` or `}` delimiters."""

    def check(self, ctx: ValidationContext) -> tuple[tuple[str, ...], tuple[str, ...]]:
        orig_delta, _ = _scan_delimiter_deltas(ctx.orig_clean)
        patched_delta, _ = _scan_delimiter_deltas(ctx.patched_clean)
        if abs(patched_delta) > abs(orig_delta):
            direction = "unclosed `{`" if patched_delta > 0 else "extra `}`"
            return (f"Unbalanced braces introduced ({direction}).",), ()
        return (), ()


class DisplayMathDelimiterRule:
    """Reject patches introducing net new unbalanced `\\[` or `\\]` display math delimiters."""

    def check(self, ctx: ValidationContext) -> tuple[tuple[str, ...], tuple[str, ...]]:
        _, orig_math = _scan_delimiter_deltas(ctx.orig_clean)
        _, patched_math = _scan_delimiter_deltas(ctx.patched_clean)
        if abs(patched_math) > abs(orig_math):
            direction = "unclosed `\\[`" if patched_math > 0 else "extra `\\]`"
            return (f"Unbalanced display math delimiters introduced ({direction}).",), ()
        return (), ()


class PreambleOrderRule:
    """Reject patches introducing `\\usepackage` before `\\documentclass`."""

    def check(self, ctx: ValidationContext) -> tuple[tuple[str, ...], tuple[str, ...]]:
        orig_before = _count_usepackages_before_documentclass(ctx.orig_clean)
        patched_before = _count_usepackages_before_documentclass(ctx.patched_clean)
        if patched_before > orig_before:
            msg = (
                "Invalid LaTeX preamble order: `\\usepackage` cannot appear before "
                "`\\documentclass`. Place package imports after `\\documentclass{...}` "
                "and before `\\begin{document}` (or use `kind='insert_preamble'`)."
            )
            return (msg,), ()
        return (), ()


def _count_usepackages_before_documentclass(clean_tex: str) -> int:
    doc_match = _DOC_CLASS_RE.search(clean_tex)
    if doc_match is None:
        return 0
    return sum(1 for _ in _USEPACKAGE_RE.finditer(clean_tex[: doc_match.start()]))


class CitationKeyVerificationRule:
    """Warn if a patch introduces new `\\cite{...}` keys absent from project `.bib` files."""

    def check(self, ctx: ValidationContext) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if not ctx.known_cite_keys:
            return (), ()
        orig_cites = set(extract_cite_keys(ctx.orig_clean))
        patched_cites = set(extract_cite_keys(ctx.patched_clean))
        new_cites = sorted(k for k in (patched_cites - orig_cites) if k not in ctx.known_cite_keys)
        if not new_cites:
            return (), ()
        warn = f"Citation key(s) not verified in project .bib files: {', '.join(new_cites[:5])}."
        return (), (warn,)


class ReferenceLabelVerificationRule:
    """Warn if a patch references `\\label{...}` keys absent from the project."""

    def check(self, ctx: ValidationContext) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if not ctx.known_labels:
            return (), ()
        orig_refs = set(extract_ref_uses(ctx.orig_clean))
        patched_refs = set(extract_ref_uses(ctx.patched_clean))
        available = set(ctx.known_labels) | set(extract_label_defs(ctx.patched_clean))
        missing = sorted(r for r in (patched_refs - orig_refs) if r not in available)
        if not missing:
            return (), ()
        warn = f"Reference label(s) not found in known page labels: {', '.join(missing[:5])}."
        return (), (warn,)


DEFAULT_VALIDATION_RULES: tuple[LatexValidationRule, ...] = (
    PreambleOrderRule(),
    EnvironmentBalanceRule(),
    BraceBalanceRule(),
    DisplayMathDelimiterRule(),
    CitationKeyVerificationRule(),
    ReferenceLabelVerificationRule(),
)


def _prepare_clean_tex(latex: str) -> str:
    return "\n".join(mask_environments(latex, skipped_envs=VERBATIM_ENVS, strip_inline_verb=True))


def validate_patched_latex(
    original_content: str,
    patched_content: str,
    *,
    known_cite_keys: frozenset[str] = frozenset(),
    known_labels: frozenset[str] = frozenset(),
    rules: Sequence[LatexValidationRule] = DEFAULT_VALIDATION_RULES,
) -> PatchValidationResult:
    """Verify patched LaTeX across the composable rule pipeline without invoking any LLM."""
    ctx = ValidationContext(
        orig_clean=_prepare_clean_tex(original_content),
        patched_clean=_prepare_clean_tex(patched_content),
        known_cite_keys=known_cite_keys,
        known_labels=known_labels,
    )

    errors: list[str] = []
    warnings: list[str] = []
    for rule in rules:
        rule_errors, rule_warnings = rule.check(ctx)
        errors.extend(rule_errors)
        warnings.extend(rule_warnings)

    return PatchValidationResult(
        ok=len(errors) == 0,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
