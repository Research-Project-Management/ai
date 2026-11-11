"""Pure immutable contracts for verified anchor-based LaTeX patches."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PatchKind = Literal["replace", "insert_preamble", "insert_after", "insert_before"]
MatchStrategy = Literal["exact", "normalized_whitespace", "preamble"]


class EditorPatchProposal(BaseModel):
    """Anchor-verified LaTeX edit ready for frontend Diff Card & CodeMirror 6 review."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., min_length=1)
    file_id: str | None = None
    file_path: str = Field(..., min_length=1)
    kind: PatchKind = "replace"
    title: str = Field(..., min_length=1)
    rationale: str = ""
    search_text: str
    replace_text: str
    from_line: int = Field(..., ge=1)
    to_line: int = Field(..., ge=1)
    match_strategy: MatchStrategy = "exact"
    warnings: tuple[str, ...] = ()


class PatchValidationResult(BaseModel):
    """Structural and symbol verification outcome for a candidate LaTeX patch."""

    model_config = ConfigDict(frozen=True)

    ok: bool = True
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
