"""Canonical inline citation formatting, prompt rules, and provenance mapping."""

import re
from collections.abc import Sequence
from typing import Final

from ai.orchestrator.contracts import CitationItem

CITATION_INSTRUCTIONS: Final[str] = (
    "Cite supporting evidence inline immediately after a claim using the exact marker syntax "
    "[[cite:N]] matching the [cite:N] labels below (for example [[cite:1]] or [[cite:1]][[cite:2]]). "
    "Cite sparingly: place one [[cite:N]] at the end of a paragraph or bullet group that shares "
    "the same source and page rather than repeating the same marker on every consecutive sub-bullet; "
    "only repeat when switching to a different source/page or highlighting an independent metric. "
    "Never use plain [N] brackets for evidence citations, and never group multiple numbers inside "
    "one marker like [[cite:1, 2]]. Never mention internal passage, chunk, or reference numbers in "
    "your prose (for example never say 'passages [1], [3] only list references'). Silently ignore "
    "irrelevant excerpts; if a detail is missing from the evidence, state naturally what is not "
    "covered without citing passage indices."
)

_CANONICAL_CITE_RE: Final[re.Pattern[str]] = re.compile(
    r"\[\[cite:\s*(\d+(?:\s*,\s*\d+)*)\s*\]\]",
    re.IGNORECASE,
)
_LEGACY_CITE_RE: Final[re.Pattern[str]] = re.compile(r"(?<!\[)\[(\d+)\](?!\()")
_MULTI_SPACE_RE: Final[re.Pattern[str]] = re.compile(r"[ \t]{2,}")


def format_evidence_tag(index: int) -> str:
    """Format the numbered evidence block prefix injected into synthesis prompts."""
    return f"[cite:{index}]"


def sanitize_citation_markers(
    answer_text: str,
    *,
    valid_indices: frozenset[int],
) -> tuple[str, int, int]:
    """Normalize grouped `[[cite:1, 2]]` tags and strip phantom/out-of-bounds `[[cite:N]]` markers.

    Returns `(cleaned_text, valid_marker_count, stripped_phantom_count)`.
    """
    if not answer_text:
        return answer_text, 0, 0

    valid_count = 0
    phantom_count = 0

    def _replace_match(match: re.Match[str]) -> str:
        nonlocal valid_count, phantom_count
        kept_tags: list[str] = []
        for raw_num in match.group(1).split(","):
            num_str = raw_num.strip()
            if not num_str.isdigit():
                continue
            idx = int(num_str)
            if idx in valid_indices:
                valid_count += 1
                kept_tags.append(f"[[cite:{idx}]]")
            else:
                phantom_count += 1
        return "".join(kept_tags)

    cleaned = _CANONICAL_CITE_RE.sub(_replace_match, answer_text)
    if phantom_count > 0:
        cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
        cleaned = re.sub(r" +([.,;!?])", r"\1", cleaned)
    return cleaned, valid_count, phantom_count


def extract_cited_indices(answer_text: str, *, max_index: int) -> list[int]:
    """Extract unique 1-based cited indices in first-appearance order.

    Prefers canonical ``[[cite:N]]`` markers so paper bibliography numbers such as ``[9]``
    are never misidentified. Falls back to standalone ``[N]`` only when no canonical marker
    appears anywhere in ``answer_text``.
    """
    if max_index <= 0 or not answer_text:
        return []

    cited: list[int] = []
    seen: set[int] = set()

    canonical_matches = list(_CANONICAL_CITE_RE.finditer(answer_text))
    if canonical_matches:
        for match in canonical_matches:
            for raw_num in match.group(1).split(","):
                idx = int(raw_num.strip())
                if 1 <= idx <= max_index and idx not in seen:
                    seen.add(idx)
                    cited.append(idx)
        return cited

    for match in _LEGACY_CITE_RE.finditer(answer_text):
        idx = int(match.group(1))
        if 1 <= idx <= max_index and idx not in seen:
            seen.add(idx)
            cited.append(idx)
    return cited


def select_indexed_citations(
    answer_text: str,
    candidates: Sequence[CitationItem],
) -> list[CitationItem]:
    """Map inline citations to canonical indexed ``CitationItem`` entries."""
    if not candidates:
        return []

    cited_indices = extract_cited_indices(answer_text, max_index=len(candidates))
    if not cited_indices:
        return []

    return [candidates[idx - 1].model_copy(update={"index": idx}) for idx in cited_indices]
