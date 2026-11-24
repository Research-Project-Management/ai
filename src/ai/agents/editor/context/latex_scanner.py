"""Shared deterministic LaTeX scanning, masking, and symbol extraction helpers."""

from __future__ import annotations

import posixpath
import re

VERBATIM_ENVS: frozenset[str] = frozenset({"verbatim", "lstlisting", "minted", "comment"})

_BEGIN_ENV_RE = re.compile(r"\\begin\{([^{}\s]+)\}")
_END_ENV_RE = re.compile(r"\\end\{([^{}\s]+)\}")
_VERB_INLINE_RE = re.compile(r"\\verb\*?(.)(?:(?!\1).)*\1")
_CITE_KEYS_RE = re.compile(r"\\(?:[a-zA-Z]*cite[a-zA-Z]*)\*?(?:\[[^\]]*\])*\{([^}]+)\}")
_LABEL_DEF_RE = re.compile(r"\\label\{([^}]+)\}")
_REF_USE_RE = re.compile(r"\\(?:eq|auto|c|C|page|name)?ref\*?\{([^}]+)\}")


def normalize_newlines(text: str) -> str:
    """Normalize CRLF/CR line endings to LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_path(path: str | None) -> str:
    """Normalize a LaTeX workspace file path to a clean relative POSIX path."""
    clean = (path or "").strip().replace("\\", "/").lstrip("/")
    while clean.startswith("./"):
        clean = clean[2:]
    return posixpath.normpath(clean) if clean else ""


def is_tex_or_sty_path(path: str, *, is_root: bool = False) -> bool:
    """Return True if `path` is a `.tex`/`.sty` (or extensionless) source file and not `.bib`."""
    lower = path.lower()
    if lower.endswith(".bib"):
        return False
    filename = path.rsplit("/", 1)[-1]
    return is_root or lower.endswith((".tex", ".sty")) or "." not in filename


def strip_latex_comments(text: str) -> str:
    """Strip unescaped `%` LaTeX comments line by line using backslash parity."""
    if not text:
        return ""
    norm = normalize_newlines(text)
    cleaned_lines: list[str] = []
    for line in norm.split("\n"):
        out_chars: list[str] = []
        backslashes = 0
        for ch in line:
            if ch == "%":
                if backslashes % 2 == 0:
                    break
                out_chars.append(ch)
                backslashes = 0
            elif ch == "\\":
                out_chars.append(ch)
                backslashes += 1
            else:
                out_chars.append(ch)
                backslashes = 0
        cleaned_lines.append("".join(out_chars))
    return "\n".join(cleaned_lines)


def mask_environments(
    content: str,
    *,
    skipped_envs: frozenset[str] = VERBATIM_ENVS,
    strip_inline_verb: bool = False,
) -> list[str]:
    """Return 1-to-1 line list with comments stripped and `skipped_envs` interiors masked to `""`."""
    if not content:
        return []

    stripped = strip_latex_comments(content)
    raw_lines = stripped.splitlines()
    masked: list[str] = []
    env_stack: list[str] = []

    for line in raw_lines:
        clean = _VERB_INLINE_RE.sub("", line) if strip_inline_verb else line
        b_matches = _BEGIN_ENV_RE.findall(clean)
        e_matches = _END_ENV_RE.findall(clean)
        inside_before = any(env in skipped_envs for env in env_stack)

        if b_matches or e_matches:
            env_stack.extend(b_matches)
            for env in e_matches:
                if env in env_stack:
                    rev_idx = len(env_stack) - 1 - env_stack[::-1].index(env)
                    env_stack.pop(rev_idx)
            inside_now = any(env in skipped_envs for env in env_stack)
            if inside_before and inside_now:
                masked.append("")
            else:
                # Collapse any single-line `\begin{skipped}...\end{skipped}` body safely
                for env in b_matches:
                    if env in skipped_envs and env in e_matches:
                        clean = re.sub(
                            rf"\\begin\{{{re.escape(env)}\}}.*?\\end\{{{re.escape(env)}\}}",
                            rf"\\begin{{{env}}}\\end{{{env}}}",
                            clean,
                        )
                masked.append(clean)
            continue

        masked.append("" if inside_before else clean)

    return masked


def extract_cite_keys(clean_tex: str) -> tuple[str, ...]:
    """Extract ordered unique citation keys from `\\cite*{...}` macros in cleaned LaTeX."""
    seen: dict[str, None] = {}
    for match in _CITE_KEYS_RE.finditer(clean_tex):
        for part in match.group(1).split(","):
            key = part.strip()
            if key:
                seen.setdefault(key, None)
    return tuple(seen.keys())


def extract_label_defs(clean_tex: str) -> tuple[str, ...]:
    """Extract ordered unique `\\label{...}` keys in cleaned LaTeX."""
    seen: dict[str, None] = {}
    for match in _LABEL_DEF_RE.finditer(clean_tex):
        key = match.group(1).strip()
        if key:
            seen.setdefault(key, None)
    return tuple(seen.keys())


def extract_ref_uses(clean_tex: str) -> tuple[str, ...]:
    """Extract ordered unique referenced label keys across `\\ref`/`\\eqref`/`\\cref` etc."""
    seen: dict[str, None] = {}
    for match in _REF_USE_RE.finditer(clean_tex):
        for part in match.group(1).split(","):
            key = part.strip()
            if key:
                seen.setdefault(key, None)
    return tuple(seen.keys())
