"""Rule pipeline for deterministic LaTeX patch validation."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

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
    orig_brace_delta: int = 0
    orig_math_delta: int = 0
    patched_brace_delta: int = 0
    patched_math_delta: int = 0


type LatexValidationRule = Callable[[ValidationContext], tuple[tuple[str, ...], tuple[str, ...]]]


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


def _count_usepackages_before_documentclass(clean_tex: str) -> int:
    doc_match = _DOC_CLASS_RE.search(clean_tex)
    if doc_match is None:
        return 0
    return sum(1 for _ in _USEPACKAGE_RE.finditer(clean_tex[: doc_match.start()]))


def check_environment_balance(ctx: ValidationContext) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Reject patches introducing net new `\\begin{env}` / `\\end{env}` mismatches."""
    orig_mismatches, _ = _count_env_mismatches(ctx.orig_clean)
    patched_mismatches, notes = _count_env_mismatches(ctx.patched_clean)
    if patched_mismatches > orig_mismatches:
        detail = "; ".join(notes[:3]) or "environment mismatch"
        return (f"Unbalanced LaTeX environment introduced ({detail}).",), ()
    return (), ()


def check_brace_balance(ctx: ValidationContext) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Reject patches introducing net new unbalanced `{` or `}` delimiters."""
    if abs(ctx.patched_brace_delta) > abs(ctx.orig_brace_delta):
        direction = "unclosed `{`" if ctx.patched_brace_delta > 0 else "extra `}`"
        return (f"Unbalanced braces introduced ({direction}).",), ()
    return (), ()


def check_display_math_delimiters(
    ctx: ValidationContext,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Reject patches introducing net new unbalanced `\\[` or `\\]` display math delimiters."""
    if abs(ctx.patched_math_delta) > abs(ctx.orig_math_delta):
        direction = "unclosed `\\[`" if ctx.patched_math_delta > 0 else "extra `\\]`"
        return (f"Unbalanced display math delimiters introduced ({direction}).",), ()
    return (), ()


def check_preamble_order(ctx: ValidationContext) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Reject patches introducing `\\usepackage` before `\\documentclass`."""
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


def check_citation_keys(ctx: ValidationContext) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Warn if a patch introduces new `\\cite{...}` keys absent from project `.bib` files."""
    if not ctx.known_cite_keys:
        return (), ()
    orig_cites = set(extract_cite_keys(ctx.orig_clean))
    patched_cites = set(extract_cite_keys(ctx.patched_clean))
    new_cites = sorted(k for k in (patched_cites - orig_cites) if k not in ctx.known_cite_keys)
    if not new_cites:
        return (), ()
    warn = f"Citation key(s) not verified in project .bib files: {', '.join(new_cites[:5])}."
    return (), (warn,)


def check_reference_labels(ctx: ValidationContext) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Warn if a patch references `\\label{...}` keys absent from the project."""
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
    check_preamble_order,
    check_environment_balance,
    check_brace_balance,
    check_display_math_delimiters,
    check_citation_keys,
    check_reference_labels,
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
    orig_clean = _prepare_clean_tex(original_content)
    patched_clean = _prepare_clean_tex(patched_content)
    orig_brace_delta, orig_math_delta = _scan_delimiter_deltas(orig_clean)
    patched_brace_delta, patched_math_delta = _scan_delimiter_deltas(patched_clean)
    ctx = ValidationContext(
        orig_clean=orig_clean,
        patched_clean=patched_clean,
        known_cite_keys=known_cite_keys,
        known_labels=known_labels,
        orig_brace_delta=orig_brace_delta,
        orig_math_delta=orig_math_delta,
        patched_brace_delta=patched_brace_delta,
        patched_math_delta=patched_math_delta,
    )

    errors: list[str] = []
    warnings: list[str] = []
    for rule in rules:
        rule_errors, rule_warnings = rule(ctx)
        errors.extend(rule_errors)
        warnings.extend(rule_warnings)

    return PatchValidationResult(
        ok=len(errors) == 0,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
