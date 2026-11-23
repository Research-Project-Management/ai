"""Contracts and domain DTOs for the Flux Pages/Editor Backend Gateway."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class EditorScope(BaseModel):
    """Scoped coordinates for the active LaTeX page project and open file."""

    model_config = ConfigDict(frozen=True)

    page_id: str = ""
    space_id: str = ""
    user_id: str = ""
    active_file_id: str | None = None
    active_file_path: str | None = None


class EditorGatewayError(RuntimeError):
    """Structured error raised when a downstream Backend Pages call fails."""

    def __init__(self, status_code: int | None, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class EditorFileInfo(BaseModel):
    """Metadata (and optional hydrated content) of one file in a LaTeX project."""

    model_config = ConfigDict(frozen=True)

    id: str
    path: str
    title: str = ""
    is_root: bool = False
    content: str | None = None


class EditorDocument(BaseModel):
    """Hydrated content and revision metadata for a single LaTeX/BibTeX document."""

    model_config = ConfigDict(frozen=True)

    id: str
    path: str
    title: str = ""
    content: str = ""
    rev: int | None = None
    line_count: int = 0


class EditorSearchHit(BaseModel):
    """Single full-text/regex match inside a page file."""

    model_config = ConfigDict(frozen=True)

    file_id: str = ""
    file_path: str
    line: int
    column: int | None = None
    preview: str


class EditorCitationEntry(BaseModel):
    """Normalized BibTeX citation entry returned by the page citations service."""

    model_config = ConfigDict(frozen=True)

    key: str
    entry_type: str = ""
    title: str = ""
    authors: str = ""
    year: str = ""
    file_path: str = ""
    raw: str = ""


class EditorSymbolCatalog(BaseModel):
    """Unified project symbol catalog (labels, citations, packages, commands, outline, files)."""

    model_config = ConfigDict(frozen=True)

    root_doc_path: str | None = None
    labels: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    packages: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()
    environments: tuple[str, ...] = ()
    outline: list[dict[str, Any]] = Field(default_factory=list)
    files: list[dict[str, Any]] = Field(default_factory=list)


class EditorReviewComment(BaseModel):
    """Inline collaborator review comment thread in a page."""

    model_config = ConfigDict(frozen=True)

    id: str
    status: str = "open"
    author: str = ""
    content: str
    line: int | None = None
    line_end: int | None = None
    replies: tuple[dict[str, Any], ...] = ()


class EditorReviewSuggestion(BaseModel):
    """Pending Track Changes suggestion in a page."""

    model_config = ConfigDict(frozen=True)

    id: str
    status: str = "pending"
    change_type: str = "insert"
    author: str = ""
    original_text: str = ""
    suggested_text: str = ""
    from_line: int | None = None
    to_line: int | None = None


class EditorGateway(Protocol):
    """Zero-trust protocol between `editor_agent` and Flux Pages Backend."""

    async def list_files(
        self,
        scope: EditorScope,
        *,
        include_content: bool = False,
    ) -> list[EditorFileInfo]: ...

    async def read_file(self, scope: EditorScope, doc_id: str) -> EditorDocument: ...

    async def get_symbols(self, scope: EditorScope) -> EditorSymbolCatalog: ...

    async def search_pages(
        self,
        scope: EditorScope,
        *,
        query: str,
        case_sensitive: bool = False,
        whole_word: bool = False,
        use_regex: bool = False,
        file_ids: Sequence[str] = (),
        max_results: int = 30,
    ) -> list[EditorSearchHit]: ...

    async def list_citations(
        self,
        scope: EditorScope,
        *,
        query: str = "",
        limit: int = 100,
    ) -> list[EditorCitationEntry]: ...

    async def validate_citations(self, scope: EditorScope) -> dict[str, Any]: ...

    async def lint_pages(
        self,
        scope: EditorScope,
        *,
        content: str | None = None,
        file_path: str | None = None,
    ) -> dict[str, Any]: ...

    async def list_review_comments(
        self,
        scope: EditorScope,
        *,
        doc_id: str | None = None,
        status: str | None = None,
    ) -> list[EditorReviewComment]: ...

    async def list_review_suggestions(
        self,
        scope: EditorScope,
        *,
        doc_id: str | None = None,
        status: str | None = None,
    ) -> list[EditorReviewSuggestion]: ...

    async def add_bibtex_entries(
        self,
        scope: EditorScope,
        *,
        raw_bibtex: str | None = None,
        entries: Sequence[dict[str, Any]] = (),
        target_bib_file: str = "references.bib",
    ) -> dict[str, Any]: ...

    async def resolve_citation(
        self,
        scope: EditorScope,
        *,
        identifier: str,
        target_bib_file: str = "references.bib",
    ) -> dict[str, Any]: ...

    async def sync_library_citations(
        self,
        scope: EditorScope,
        *,
        collection_id: str,
        target_bib_file: str = "references.bib",
    ) -> dict[str, Any]: ...

    async def create_review_suggestion(
        self,
        scope: EditorScope,
        *,
        doc_id: str,
        change_type: Literal["insert", "delete"],
        suggested_text: str = "",
        original_text: str = "",
        from_line: int,
        from_column: int,
        to_line: int,
        to_column: int,
    ) -> dict[str, Any]: ...

    async def create_page(
        self,
        scope: EditorScope,
        *,
        title: str,
        description: str | None = None,
        content: str | None = None,
        template_type: str | None = None,
    ) -> dict[str, Any]: ...
