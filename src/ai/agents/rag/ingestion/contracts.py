"""Contracts for RAG document parsing, structure-aware chunking, and ingestion."""

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class UnsupportedDocumentFormatError(ValueError):
    """Raised when an inbound file matches no registered DocumentParser."""


class ParsedPage(BaseModel):
    """One parsed page or logical section represented as clean Markdown."""

    model_config = ConfigDict(frozen=True)

    page_number: int | None = None
    markdown: str


class ParsedDocument(BaseModel):
    """Normalized parser output independent of underlying binary format."""

    model_config = ConfigDict(frozen=True)

    title: str = ""
    pages: tuple[ParsedPage, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class DocumentParser(Protocol):
    """Port interface for converting raw file bytes into structured Markdown pages."""

    @property
    def supported_extensions(self) -> frozenset[str]:
        """Lowercase dotted extensions supported by this parser (e.g. `.pdf`)."""
        ...

    @property
    def supported_mime_types(self) -> frozenset[str]:
        """Normalized lowercase MIME types supported by this parser."""
        ...

    def parse_bytes(self, data: bytes, *, filename: str = "") -> ParsedDocument:
        """Parse binary document bytes into structured Markdown pages."""
        ...


class IngestDocumentRequest(BaseModel):
    """Immutable command describing a raw file or text document to ingest into a Space."""

    model_config = ConfigDict(frozen=True)

    space_id: str = Field(..., min_length=1)
    source_type: str = Field(..., min_length=1)
    source_id: str = Field(..., min_length=1)
    title: str = ""
    filename: str = ""
    mime_type: str = ""
    raw_bytes: bytes | None = None
    content: str = ""
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResult(BaseModel):
    """Immutable result summary returned after parsing, chunking, and upserting."""

    model_config = ConfigDict(frozen=True)

    space_id: str
    source_type: str
    source_id: str
    title: str
    pages_parsed: int
    chunks_indexed: int
