"""Library read client: search, overview, item detail, health audit, citations."""

import asyncio
from collections.abc import Sequence
from typing import Any, Final, Literal
from urllib.parse import quote

from ai.agents.library.gateway.client_base import library_route
from ai.agents.library.gateway.contracts import (
    ItemDetail,
    LibraryGatewayError,
    LibraryOverview,
    LibraryScope,
    TagSummary,
)
from ai.agents.library.gateway.mapping import (
    normalize_read_status,
    project_collections,
    project_summary,
)
from ai.shared.gateway.transport import BackendTransport

_EXPORT_FORMATS: Final[frozenset[str]] = frozenset(
    {"bibtex", "ris", "csl-json", "csl_json", "csv", "markdown"}
)
_MAX_OVERVIEW_TAGS: Final[int] = 50
_MAX_ITEM_NOTES: Final[int] = 10
_MAX_NOTE_CONTENT_CHARS: Final[int] = 600
_MAX_ITEM_RELATIONS: Final[int] = 15
_MAX_ABSTRACT_CHARS: Final[int] = 1500
_MAX_AUDIT_ENTRIES: Final[int] = 20
_MAX_CITE_BATCH_IDS: Final[int] = 100


class LibraryReader:
    """`LibraryReadPort` over the Flux Backend Library REST API."""

    def __init__(self, transport: BackendTransport, *, max_items_per_call: int = 20) -> None:
        self._transport = transport
        self._max_items_per_call = max_items_per_call

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
    ) -> dict[str, Any]:
        capped = max(1, min(limit, self._max_items_per_call))
        clean_query = query.strip()
        params: dict[str, Any] = {
            "fromYear": year_from,
            "toYear": year_to,
            "itemType": item_type,
            "collectionId": collection_id,
            "tag": tag,
            "readStatus": normalize_read_status(read_status),
            "limit": capped,
        }
        if clean_query:
            params["q"] = clean_query
            raw = await self._transport.request(
                "GET", library_route(scope, "search"), params=params
            )
        else:
            raw = await self._transport.request("GET", library_route(scope, "items"), params=params)

        raw_items = raw.get("items", []) if isinstance(raw, dict) else raw
        items = [
            project_summary(item).model_dump(mode="json")
            for item in (raw_items if isinstance(raw_items, list) else [])[:capped]
            if isinstance(item, dict)
        ]
        meta = raw.get("meta") or raw.get("pagination") or {} if isinstance(raw, dict) else {}
        return {
            "items": items,
            "returned_count": len(items),
            "total_count": meta.get("totalCount", len(items))
            if isinstance(meta, dict)
            else len(items),
        }

    async def get_overview(self, scope: LibraryScope) -> LibraryOverview:
        counts_raw, tree_raw, tags_raw = await asyncio.gather(
            self._transport.request("GET", library_route(scope, "items/counts")),
            self._transport.request("GET", library_route(scope, "collections/tree")),
            self._transport.request("GET", library_route(scope, "tags")),
        )
        counts = counts_raw if isinstance(counts_raw, dict) else {}
        tree_list = (
            tree_raw.get("tree") or tree_raw.get("collections") or []
            if isinstance(tree_raw, dict)
            else tree_raw
        )
        raw_tags = tags_raw.get("tags", []) if isinstance(tags_raw, dict) else tags_raw
        tags: list[TagSummary] = []
        if isinstance(raw_tags, list):
            for entry in raw_tags[:_MAX_OVERVIEW_TAGS]:
                if isinstance(entry, dict) and str(entry.get("name") or "").strip():
                    tags.append(
                        TagSummary(
                            id=str(entry.get("id") or ""),
                            name=str(entry.get("name") or "").strip(),
                            count=int(entry.get("itemCount") or entry.get("count") or 0),
                        )
                    )
        return LibraryOverview(
            counts=counts,
            collections=project_collections(tree_list),
            tags=tuple(tags),
        )

    async def get_item(
        self,
        scope: LibraryScope,
        item_id: str,
        *,
        include_notes: bool = True,
        include_relations: bool = True,
    ) -> ItemDetail:
        enc_id = quote(item_id.strip(), safe="")
        item_raw = await self._transport.request("GET", library_route(scope, f"items/{enc_id}"))
        if not isinstance(item_raw, dict):
            raise LibraryGatewayError(502, "Unexpected item response format from Backend")

        summary = project_summary(item_raw)
        notes: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []

        async def _safe_get(suffix: str, params: dict[str, Any] | None = None) -> Any:
            try:
                return await self._transport.request(
                    "GET", library_route(scope, suffix), params=params
                )
            except LibraryGatewayError:
                return None

        aux_tasks: list[Any] = []
        aux_tasks.append(_safe_get("notes", {"itemId": item_id}) if include_notes else None)
        aux_tasks.append(_safe_get(f"items/{enc_id}/relations") if include_relations else None)
        resolved = await asyncio.gather(*(t for t in aux_tasks if t is not None))

        idx = 0
        if include_notes:
            notes_raw = resolved[idx]
            idx += 1
            raw_note_list = (
                notes_raw.get("notes", notes_raw) if isinstance(notes_raw, dict) else notes_raw
            )
            if isinstance(raw_note_list, list):
                for note in raw_note_list[:_MAX_ITEM_NOTES]:
                    if isinstance(note, dict):
                        notes.append(
                            {
                                "id": str(note.get("id") or ""),
                                "title": str(note.get("title") or ""),
                                "content": str(note.get("contentMd") or note.get("content") or "")[
                                    :_MAX_NOTE_CONTENT_CHARS
                                ],
                            }
                        )
        if include_relations:
            rel_raw = resolved[idx]
            raw_rel_list = (
                rel_raw.get("relations", rel_raw) if isinstance(rel_raw, dict) else rel_raw
            )
            if isinstance(raw_rel_list, list):
                for rel in raw_rel_list[:_MAX_ITEM_RELATIONS]:
                    if isinstance(rel, dict):
                        relations.append(
                            {
                                "id": str(rel.get("id") or rel.get("targetItemId") or ""),
                                "title": str(rel.get("title") or ""),
                                "relation_type": str(rel.get("relationType") or "related"),
                            }
                        )

        abstract = str(item_raw.get("abstract") or item_raw.get("abstractNote") or "").strip()[
            :_MAX_ABSTRACT_CHARS
        ]
        attachments = item_raw.get("attachments")
        attachment_count = (
            len(attachments)
            if isinstance(attachments, list)
            else int(item_raw.get("attachmentCount") or 0)
        )
        return ItemDetail(
            **summary.model_dump(),
            abstract=abstract,
            citation_key=str(item_raw.get("citationKey") or item_raw.get("citeKey") or ""),
            notes=tuple(notes),
            relations=tuple(relations),
            attachment_count=attachment_count,
        )

    async def audit_health(
        self,
        scope: LibraryScope,
        *,
        check: Literal["all", "duplicates", "quality", "retractions"] = "all",
    ) -> dict[str, Any]:
        tasks: dict[str, Any] = {}
        if check in ("all", "duplicates"):
            tasks["duplicates"] = self._transport.request(
                "GET", library_route(scope, "curation/duplicates")
            )
        if check in ("all", "quality"):
            tasks["quality"] = self._transport.request(
                "GET", library_route(scope, "curation/quality")
            )
        if check in ("all", "retractions"):
            tasks["retractions"] = self._transport.request(
                "GET", library_route(scope, "retraction/items")
            )

        keys = list(tasks.keys())
        values = await asyncio.gather(*(tasks[k] for k in keys))
        report: dict[str, Any] = {}
        for key, val in zip(keys, values, strict=True):
            if isinstance(val, list):
                report[key] = val[:_MAX_AUDIT_ENTRIES]
            elif isinstance(val, dict):
                compact: dict[str, Any] = {}
                for sub_k, sub_v in val.items():
                    compact[sub_k] = (
                        sub_v[:_MAX_AUDIT_ENTRIES] if isinstance(sub_v, list) else sub_v
                    )
                report[key] = compact
            else:
                report[key] = val
        return report

    async def cite_items(
        self,
        scope: LibraryScope,
        *,
        item_ids: Sequence[str] = (),
        style: str = "apa-7th",
        export_format: str | None = None,
    ) -> dict[str, Any]:
        fmt = (export_format or "").strip().lower()
        clean_ids = [str(i).strip() for i in item_ids if str(i).strip()][:_MAX_CITE_BATCH_IDS]

        if fmt in _EXPORT_FORMATS:
            if clean_ids:
                raw = await self._transport.request(
                    "POST",
                    library_route(scope, "exports"),
                    json_body={"format": fmt, "itemIds": clean_ids},
                )
            else:
                raw = await self._transport.request(
                    "GET",
                    library_route(scope, "exports"),
                    params={"format": fmt},
                )
            if isinstance(raw, dict):
                return {
                    "format": fmt,
                    "filename": raw.get("filename"),
                    "item_count": raw.get("itemCount") or raw.get("total"),
                    "content": raw.get("content") or raw.get("bibtex") or raw.get("text") or "",
                }
            return {"format": fmt, "content": str(raw)}

        if not clean_ids:
            raise LibraryGatewayError(
                400,
                "cite_items requires item_ids when formatting styles (or pass export_format for bulk export)",
            )
        raw = await self._transport.request(
            "POST",
            library_route(scope, "citation/batch-items"),
            json_body={"itemIds": clean_ids, "style": style.strip() or "apa-7th"},
        )
        return raw if isinstance(raw, dict) else {"citations": raw}
