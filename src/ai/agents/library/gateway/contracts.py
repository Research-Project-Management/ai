"""Gateway protocol, compact DTOs, and scope contracts for `library_agent`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class LibraryGatewayError(RuntimeError):
    """Typed HTTP/gateway error mapped for agent consumption without leaking secrets."""

    def __init__(self, status_code: int | None, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class LibraryScope(BaseModel):
    """Workspace scope routed to personal or project library Backend endpoints."""

    model_config = ConfigDict(frozen=True)

    space_id: str = ""
    user_id: str = ""
    scope_type: Literal["auto", "personal", "project"] = "auto"


class ItemSummary(BaseModel):
    """Compact bibliographic summary projected to keep LLM context bounded."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str = "Untitled"
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str = ""
    item_type: str = ""
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    read_status: str = ""
    tags: tuple[str, ...] = ()
    collections: tuple[str, ...] = ()


class ItemDetail(BaseModel):
    """Full item detail projection with truncated abstract, notes, and relations."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str = "Untitled"
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str = ""
    item_type: str = ""
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    read_status: str = ""
    tags: tuple[str, ...] = ()
    collections: tuple[str, ...] = ()
    abstract: str = ""
    citation_key: str = ""
    notes: tuple[dict[str, Any], ...] = ()
    relations: tuple[dict[str, Any], ...] = ()
    attachment_count: int = 0


class CollectionNode(BaseModel):
    """Hierarchical library collection node."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    parent_id: str | None = None
    item_count: int = 0
    children: tuple[CollectionNode, ...] = ()


CollectionNode.model_rebuild()


class TagSummary(BaseModel):
    """Compact tag entry with item count."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    count: int = 0


class LibraryOverview(BaseModel):
    """Aggregated library counts, collection tree, and tags."""

    model_config = ConfigDict(frozen=True)

    counts: dict[str, Any] = Field(default_factory=dict)
    collections: tuple[CollectionNode, ...] = ()
    tags: tuple[TagSummary, ...] = ()


class LibraryReadPort(Protocol):
    """Read-only Library operations: search, overview, detail, audit, citations."""

    async def search_items(
        self,
        scope: LibraryScope,
        *,
        query: str = "",
        year_from: int | None = None,
        year_to: int | None = None,
        item_type: str | None = None,
        collection_id: str | None = None,
        tag: str | None = None,
        read_status: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]: ...

    async def get_overview(self, scope: LibraryScope) -> LibraryOverview: ...

    async def get_item(
        self,
        scope: LibraryScope,
        item_id: str,
        *,
        include_notes: bool = True,
        include_relations: bool = True,
    ) -> ItemDetail: ...

    async def audit_health(
        self,
        scope: LibraryScope,
        *,
        check: Literal["all", "duplicates", "quality", "retractions"] = "all",
    ) -> dict[str, Any]: ...

    async def cite_items(
        self,
        scope: LibraryScope,
        *,
        item_ids: Sequence[str] = (),
        style: str = "apa-7th",
        export_format: str | None = None,
    ) -> dict[str, Any]: ...


class LibraryWritePort(Protocol):
    """Library mutations: import, organize, notes, trash, and merge."""

    async def add_paper(
        self,
        scope: LibraryScope,
        *,
        source: Literal["doi", "arxiv", "pmid", "isbn", "url"],
        identifier: str = "",
        url: str | None = None,
        collection_id: str | None = None,
    ) -> dict[str, Any]: ...

    async def manage_collection(
        self,
        scope: LibraryScope,
        *,
        action: Literal["create", "rename", "assign", "move", "detach"],
        collection_id: str | None = None,
        name: str | None = None,
        parent_id: str | None = None,
        item_ids: Sequence[str] = (),
        source_collection_id: str | None = None,
    ) -> dict[str, Any]: ...

    async def organize_item(
        self,
        scope: LibraryScope,
        *,
        item_id: str,
        tags: Sequence[str] | None = None,
        read_status: str | None = None,
        is_starred: bool | None = None,
        link_item_ids: Sequence[str] = (),
        unlink_item_id: str | None = None,
    ) -> dict[str, Any]: ...

    async def save_note(
        self,
        scope: LibraryScope,
        *,
        note_id: str | None = None,
        item_id: str | None = None,
        title: str | None = None,
        content_md: str = "",
        tags: Sequence[str] | None = None,
    ) -> dict[str, Any]: ...

    async def trash_items(
        self,
        scope: LibraryScope,
        *,
        item_ids: Sequence[str],
    ) -> dict[str, Any]: ...

    async def merge_duplicates(
        self,
        scope: LibraryScope,
        *,
        primary_item_id: str,
        duplicate_item_ids: Sequence[str],
    ) -> dict[str, Any]: ...


class LibraryGateway(Protocol):
    """Port abstracting Flux Backend Library REST operations, split into read and write ports."""

    @property
    def read(self) -> LibraryReadPort: ...

    @property
    def write(self) -> LibraryWritePort: ...
