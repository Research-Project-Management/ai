"""Read-only idempotent Library tools (search, overview stats, details, health, citations)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ai.agents.library.gateway.contracts import LibraryGateway, LibraryScope
from ai.agents.library.tools.contracts import LibraryToolSpec, ToolResult
from ai.orchestrator.contracts import CitationItem


class SearchItemsArgs(BaseModel):
    query: str = ""
    year_from: int | None = None
    year_to: int | None = None
    item_type: str | None = None
    collection_id: str | None = None
    tag: str | None = None
    read_status: str | None = None
    limit: int = Field(default=15, ge=1, le=50)


class LibraryStatsArgs(BaseModel):
    pass


class GetItemArgs(BaseModel):
    item_id: str = Field(..., min_length=1)
    include_notes: bool = True
    include_relations: bool = True


class AuditLibraryArgs(BaseModel):
    check: Literal["all", "duplicates", "quality", "retractions"] = "all"


class CiteItemsArgs(BaseModel):
    item_ids: list[str] = Field(default_factory=list)
    style: str = "apa-7th"
    export_format: Literal["bibtex", "ris", "csl-json", "csl_json", "csv", "markdown"] | None = None


def _item_to_citation(item: dict[str, object]) -> CitationItem | None:
    item_id = str(item.get("id") or "").strip()
    if not item_id:
        return None
    title = str(item.get("title") or "").strip()
    doi = str(item.get("doi") or "").strip()
    raw_url = str(item.get("url") or "").strip()
    url = raw_url or (f"https://doi.org/{doi}" if doi else None)
    year = item.get("year")
    venue = str(item.get("venue") or "").strip()
    snippet_parts = [p for p in (str(year) if year else "", venue) if p]
    return CitationItem(
        source_id=item_id,
        title=title,
        url=url,
        snippet=" | ".join(snippet_parts),
    )


async def _handle_search_items(
    gateway: LibraryGateway,
    scope: LibraryScope,
    args: SearchItemsArgs,
) -> ToolResult:
    data = await gateway.read.search_items(
        scope,
        query=args.query,
        year_from=args.year_from,
        year_to=args.year_to,
        item_type=args.item_type,
        collection_id=args.collection_id,
        tag=args.tag,
        read_status=args.read_status,
        limit=args.limit,
    )
    items = data.get("items") or []
    citations = tuple(
        c
        for item in (items if isinstance(items, list) else [])
        if isinstance(item, dict) and (c := _item_to_citation(item)) is not None
    )
    return ToolResult(tool="search_items", status="ok", data=data, citations=citations)


async def _handle_library_stats(
    gateway: LibraryGateway,
    scope: LibraryScope,
    args: LibraryStatsArgs,
) -> ToolResult:
    del args
    overview = await gateway.read.get_overview(scope)
    return ToolResult(tool="library_stats", status="ok", data=overview.model_dump(mode="json"))


async def _handle_get_item(
    gateway: LibraryGateway,
    scope: LibraryScope,
    args: GetItemArgs,
) -> ToolResult:
    detail = await gateway.read.get_item(
        scope,
        args.item_id,
        include_notes=args.include_notes,
        include_relations=args.include_relations,
    )
    dumped = detail.model_dump(mode="json")
    cite = _item_to_citation(dumped)
    return ToolResult(
        tool="get_item",
        status="ok",
        data=dumped,
        citations=(cite,) if cite is not None else (),
    )


async def _handle_audit_library(
    gateway: LibraryGateway,
    scope: LibraryScope,
    args: AuditLibraryArgs,
) -> ToolResult:
    report = await gateway.read.audit_health(scope, check=args.check)
    return ToolResult(tool="audit_library", status="ok", data=report)


async def _handle_cite_items(
    gateway: LibraryGateway,
    scope: LibraryScope,
    args: CiteItemsArgs,
) -> ToolResult:
    data = await gateway.read.cite_items(
        scope,
        item_ids=args.item_ids,
        style=args.style,
        export_format=args.export_format,
    )
    return ToolResult(tool="cite_items", status="ok", data=data)


def build_read_tools() -> tuple[LibraryToolSpec, ...]:
    """Return all idempotent read-only Library tools."""
    return (
        LibraryToolSpec(
            name="search_items",
            description="Search/filter papers and library items by keywords, year range, collection, tag, or read status.",
            args_schema=SearchItemsArgs,
            risk_tier="read",
            idempotent=True,
            handler=_handle_search_items,
        ),
        LibraryToolSpec(
            name="library_stats",
            description="Get real library counts, collection hierarchy tree, and available tags.",
            args_schema=LibraryStatsArgs,
            risk_tier="read",
            idempotent=True,
            handler=_handle_library_stats,
        ),
        LibraryToolSpec(
            name="get_item",
            description="Inspect full bibliographic metadata, abstract, notes, and relations of a library item by item_id.",
            args_schema=GetItemArgs,
            risk_tier="read",
            idempotent=True,
            handler=_handle_get_item,
        ),
        LibraryToolSpec(
            name="audit_library",
            description="Inspect library health: duplicate clusters, metadata quality issues, and retracted papers.",
            args_schema=AuditLibraryArgs,
            risk_tier="read",
            idempotent=True,
            handler=_handle_audit_library,
        ),
        LibraryToolSpec(
            name="cite_items",
            description="Format citations (apa-7th, ieee, vancouver, etc.) or export bibtex/ris/csl-json for library item_ids.",
            args_schema=CiteItemsArgs,
            risk_tier="read",
            idempotent=True,
            handler=_handle_cite_items,
        ),
    )
