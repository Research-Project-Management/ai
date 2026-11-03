"""Safe non-destructive write Library tools (ingest paper, collections, tags/state, notes)."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from ai.agents.library.gateway.contracts import LibraryGateway, LibraryScope
from ai.agents.library.tools.contracts import LibraryToolSpec, ToolResult


class AddPaperArgs(BaseModel):
    source: Literal["doi", "arxiv", "pmid", "isbn", "url"]
    identifier: str = ""
    url: str | None = None
    collection_id: str | None = None

    @model_validator(mode="after")
    def _require_identifier_or_url(self) -> Self:
        if not self.identifier.strip() and not (self.url or "").strip():
            raise ValueError("Either identifier or url must be provided")
        return self


class ManageCollectionArgs(BaseModel):
    action: Literal["create", "rename", "assign", "move", "detach"]
    collection_id: str | None = None
    name: str | None = None
    parent_id: str | None = None
    item_ids: list[str] = Field(default_factory=list)
    source_collection_id: str | None = None

    @model_validator(mode="after")
    def _validate_action_requirements(self) -> Self:
        if self.action == "create" and not (self.name or "").strip():
            raise ValueError("name is required when action='create'")
        if self.action == "rename" and (
            not (self.collection_id or "").strip() or not (self.name or "").strip()
        ):
            raise ValueError("collection_id and name are required when action='rename'")
        if self.action in ("assign", "move", "detach") and (
            not (self.collection_id or "").strip() or not self.item_ids
        ):
            raise ValueError(f"collection_id and item_ids are required when action={self.action!r}")
        return self


class OrganizeItemArgs(BaseModel):
    item_id: str = Field(..., min_length=1)
    tags: list[str] | None = None
    read_status: Literal["unread", "reading", "read", "completed"] | None = None
    is_starred: bool | None = None
    link_item_ids: list[str] = Field(default_factory=list)
    unlink_item_id: str | None = None

    @model_validator(mode="after")
    def _require_at_least_one_update(self) -> Self:
        if (
            self.tags is None
            and self.read_status is None
            and self.is_starred is None
            and not self.link_item_ids
            and not (self.unlink_item_id or "").strip()
        ):
            raise ValueError("Provide at least one field to update on the item")
        return self


class SaveNoteArgs(BaseModel):
    content_md: str = Field(..., min_length=1)
    item_id: str | None = None
    note_id: str | None = None
    title: str | None = None
    tags: list[str] | None = None


async def _handle_add_paper(
    gateway: LibraryGateway,
    scope: LibraryScope,
    args: AddPaperArgs,
) -> ToolResult:
    data = await gateway.write.add_paper(
        scope,
        source=args.source,
        identifier=args.identifier,
        url=args.url,
        collection_id=args.collection_id,
    )
    return ToolResult(tool="add_paper", status="ok", data=data)


async def _handle_manage_collection(
    gateway: LibraryGateway,
    scope: LibraryScope,
    args: ManageCollectionArgs,
) -> ToolResult:
    data = await gateway.write.manage_collection(
        scope,
        action=args.action,
        collection_id=args.collection_id,
        name=args.name,
        parent_id=args.parent_id,
        item_ids=args.item_ids,
        source_collection_id=args.source_collection_id,
    )
    return ToolResult(tool="manage_collection", status="ok", data=data)


async def _handle_organize_item(
    gateway: LibraryGateway,
    scope: LibraryScope,
    args: OrganizeItemArgs,
) -> ToolResult:
    data = await gateway.write.organize_item(
        scope,
        item_id=args.item_id,
        tags=args.tags,
        read_status=args.read_status,
        is_starred=args.is_starred,
        link_item_ids=args.link_item_ids,
        unlink_item_id=args.unlink_item_id,
    )
    return ToolResult(tool="organize_item", status="ok", data=data)


async def _handle_save_note(
    gateway: LibraryGateway,
    scope: LibraryScope,
    args: SaveNoteArgs,
) -> ToolResult:
    data = await gateway.write.save_note(
        scope,
        note_id=args.note_id,
        item_id=args.item_id,
        title=args.title,
        content_md=args.content_md,
        tags=args.tags,
    )
    return ToolResult(tool="save_note", status="ok", data=data)


def build_write_tools() -> tuple[LibraryToolSpec, ...]:
    """Return non-destructive safe_write Library tools."""
    return (
        LibraryToolSpec(
            name="add_paper",
            description="Ingest a paper into the Flux Library from a DOI, arXiv ID, PMID, ISBN, or URL (optionally into collection_id).",
            args_schema=AddPaperArgs,
            risk_tier="safe_write",
            idempotent=False,
            handler=_handle_add_paper,
        ),
        LibraryToolSpec(
            name="manage_collection",
            description="Create or rename a library collection, or assign/move/detach item_ids in a collection.",
            args_schema=ManageCollectionArgs,
            risk_tier="safe_write",
            idempotent=False,
            handler=_handle_manage_collection,
        ),
        LibraryToolSpec(
            name="organize_item",
            description="Update an item's tags, reading status (unread/reading/completed), starred state, or link/unlink related items.",
            args_schema=OrganizeItemArgs,
            risk_tier="safe_write",
            idempotent=False,
            handler=_handle_organize_item,
        ),
        LibraryToolSpec(
            name="save_note",
            description="Create or update a Markdown literature note in the library (optionally attached to item_id).",
            args_schema=SaveNoteArgs,
            risk_tier="safe_write",
            idempotent=False,
            handler=_handle_save_note,
        ),
    )
