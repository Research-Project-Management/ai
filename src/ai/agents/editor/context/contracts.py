"""Pure immutable contracts for hydrated LaTeX editor context snapshots."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PreambleSummary(BaseModel):
    """Extracted LaTeX documentclass, packages, custom macros, and preamble slice."""

    model_config = ConfigDict(frozen=True)

    document_class: str = ""
    begin_document_line: int | None = None
    packages: tuple[str, ...] = ()
    custom_macros: tuple[str, ...] = ()
    raw_excerpt: str = ""


class CompileErrorItem(BaseModel):
    """Normalized compiler or syntax diagnostic entry."""

    model_config = ConfigDict(frozen=True)

    line: int | None = None
    file: str = ""
    severity: str = "error"
    message: str
    context: str = ""


class EditorContextSnapshot(BaseModel):
    """Normalized, token-budgeted representation of active editor state."""

    model_config = ConfigDict(frozen=True)

    page_id: str = ""
    active_file_id: str | None = None
    active_file_path: str = "main.tex"
    total_lines: int = 0
    cursor_line: int | None = None
    cursor_column: int | None = None
    selection_text: str = ""
    selection_start_line: int | None = None
    selection_end_line: int | None = None
    current_section: str = ""
    current_environment: str = ""
    document_structure_summary: str = ""
    preamble: PreambleSummary = Field(default_factory=PreambleSummary)
    local_labels: tuple[str, ...] = ()
    local_citations: tuple[str, ...] = ()
    compile_errors: tuple[CompileErrorItem, ...] = ()
    viewport_start_line: int = 1
    viewport_end_line: int = 0
    is_truncated: bool = False
    viewport_numbered: str = ""
