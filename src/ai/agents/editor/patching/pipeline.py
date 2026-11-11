"""Anchor -> overlap -> validate pipeline turning an edit request into a verified patch."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ai.agents.editor.patching.contracts import EditorPatchProposal, PatchKind
from ai.agents.editor.patching.matcher import check_patch_overlap, resolve_patch
from ai.agents.editor.patching.validator import validate_patched_latex


@dataclass(frozen=True)
class PatchOutcome:
    """Either a verified `patch` with its `patched_content`, or a planner-readable `error`."""

    patch: EditorPatchProposal | None = None
    patched_content: str = ""
    error: str | None = None


def build_verified_patch(
    existing: Sequence[EditorPatchProposal],
    *,
    content: str,
    file_id: str | None,
    file_path: str,
    kind: PatchKind,
    search_text: str,
    replace_text: str,
    title: str,
    rationale: str = "",
    known_cite_keys: frozenset[str] = frozenset(),
    known_labels: frozenset[str] = frozenset(),
) -> PatchOutcome:
    """Resolve the anchor, reject overlaps, and verify LaTeX; warnings ride on the patch."""
    proposal, patched, match_err = resolve_patch(
        content,
        file_id=file_id,
        file_path=file_path,
        kind=kind,
        search_text=search_text,
        replace_text=replace_text,
        title=title,
        rationale=rationale,
    )
    if match_err or proposal is None:
        return PatchOutcome(error=match_err or "Failed to resolve patch anchor.")

    overlap_err = check_patch_overlap(existing, proposal)
    if overlap_err:
        return PatchOutcome(error=overlap_err)

    validation = validate_patched_latex(
        content,
        patched,
        known_cite_keys=known_cite_keys,
        known_labels=known_labels,
    )
    if not validation.ok:
        return PatchOutcome(
            error="LaTeX syntax verification failed: " + "; ".join(validation.errors)
        )

    final = (
        proposal.model_copy(update={"warnings": validation.warnings})
        if validation.warnings
        else proposal
    )
    return PatchOutcome(patch=final, patched_content=patched)
