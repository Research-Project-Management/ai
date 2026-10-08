"""Typed attached-context source contracts separating item, file, and vector IDs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator


class AttachedSource(BaseModel):
    """Canonical attached source distinguishing Library items, physical files, and Qdrant IDs."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["library", "upload", "paper"] = "upload"
    title: str = ""
    source_id: str | None = None
    source_type: Literal["upload", "file"] | None = None
    item_id: str | None = None
    file_id: str | None = None
    has_pdf: bool = False
    indexed: bool = False

    @field_validator("source_id", "item_id", "file_id", mode="before")
    @classmethod
    def _clean_optional_id(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("source_type", mode="before")
    @classmethod
    def _clean_source_type(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("title", mode="before")
    @classmethod
    def _clean_title(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @property
    def effective_source_id(self) -> str | None:
        """Canonical Qdrant `source_id`; never returns `item_id` for metadata-only items."""
        if self.source_id:
            return self.source_id
        if self.has_pdf and self.file_id:
            return self.file_id
        return None

    @property
    def effective_source_type(self) -> Literal["upload", "file"]:
        """Canonical Qdrant `source_type` corresponding to `effective_source_id`."""
        if self.source_type is not None:
            return self.source_type
        return "file" if self.kind in ("library", "paper") else "upload"


def parse_attached_sources(raw: Any, *, max_items: int = 30) -> tuple[AttachedSource, ...]:
    """Parse and deduplicate `context.attached_sources` safely."""
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        return ()

    parsed: list[AttachedSource] = []
    seen: set[tuple[str, str | None, str | None, str | None]] = set()
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        try:
            source = AttachedSource.model_validate(entry)
        except ValidationError:
            continue
        if (
            source.item_id is None
            and source.file_id is None
            and source.source_id is None
            and not source.title
        ):
            continue
        key = (source.kind, source.item_id, source.file_id, source.source_id)
        if key in seen:
            continue
        seen.add(key)
        parsed.append(source)
        if len(parsed) >= max_items:
            break
    return tuple(parsed)


def collect_rag_source_ids(
    sources: Sequence[AttachedSource],
    fallback_document_ids: Sequence[str] = (),
) -> tuple[str, ...]:
    """Collect Qdrant `source_id` filters while excluding metadata-only `item_id`s."""
    ordered: list[str] = []
    seen: set[str] = set()
    metadata_only_items: set[str] = set()

    for src in sources:
        sid = src.effective_source_id
        if sid:
            if sid not in seen:
                seen.add(sid)
                ordered.append(sid)
        elif src.item_id:
            metadata_only_items.add(src.item_id)

    for raw_doc_id in fallback_document_ids:
        doc_id = str(raw_doc_id).strip()
        if not doc_id or doc_id in metadata_only_items or doc_id in seen:
            continue
        seen.add(doc_id)
        ordered.append(doc_id)

    return tuple(ordered)
