"""Library write client: import, collections, organize, notes, trash, merge."""

from collections.abc import Sequence
from typing import Any, Literal
from urllib.parse import quote

from ai.agents.library.gateway.client_base import library_route
from ai.agents.library.gateway.contracts import (
    LibraryGatewayError,
    LibraryScope,
)
from ai.agents.library.gateway.mapping import normalize_read_status
from ai.shared.gateway.transport import BackendTransport


class LibraryWriter:
    """`LibraryWritePort` over the Flux Backend Library REST API."""

    def __init__(self, transport: BackendTransport) -> None:
        self._transport = transport

    async def add_paper(
        self,
        scope: LibraryScope,
        *,
        source: Literal["doi", "arxiv", "pmid", "isbn", "url"],
        identifier: str = "",
        url: str | None = None,
        collection_id: str | None = None,
    ) -> dict[str, Any]:
        field_by_source = {
            "doi": "doi",
            "arxiv": "arxivId",
            "pmid": "pmid",
            "isbn": "isbn",
            "url": "url",
        }
        target_field = field_by_source[source]
        target_value = (url if source == "url" and url else identifier or url or "").strip()
        if not target_value:
            raise LibraryGatewayError(
                400, f"add_paper requires a non-empty value for source={source}"
            )

        body: dict[str, Any] = {"source": source, target_field: target_value}
        if collection_id and collection_id.strip():
            body["collectionId"] = collection_id.strip()
        raw = await self._transport.request(
            "POST", library_route(scope, "ingestion"), json_body=body
        )
        return raw if isinstance(raw, dict) else {"result": raw}

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
    ) -> dict[str, Any]:
        clean_ids = [str(i).strip() for i in item_ids if str(i).strip()]
        if action == "create":
            if not (name and name.strip()):
                raise LibraryGatewayError(400, "collection name is required for action='create'")
            body: dict[str, Any] = {"name": name.strip()}
            if parent_id and parent_id.strip():
                body["parentId"] = parent_id.strip()
            raw = await self._transport.request(
                "POST", library_route(scope, "collections"), json_body=body
            )
            return raw if isinstance(raw, dict) else {"result": raw}

        if not (collection_id and collection_id.strip()):
            raise LibraryGatewayError(400, f"collection_id is required for action={action!r}")
        enc_col = quote(collection_id.strip(), safe="")

        if action == "rename":
            if not (name and name.strip()):
                raise LibraryGatewayError(400, "collection name is required for action='rename'")
            raw = await self._transport.request(
                "PUT",
                library_route(scope, f"collections/{enc_col}"),
                json_body={"name": name.strip()},
            )
            return raw if isinstance(raw, dict) else {"result": raw}

        if not clean_ids:
            raise LibraryGatewayError(400, f"item_ids must not be empty for action={action!r}")

        if action == "assign":
            raw = await self._transport.request(
                "POST",
                library_route(scope, f"collections/{enc_col}/items"),
                json_body={"itemIds": clean_ids},
            )
            return raw if isinstance(raw, dict) else {"result": raw}

        if action == "move":
            move_body: dict[str, Any] = {"itemIds": clean_ids}
            if source_collection_id and source_collection_id.strip():
                move_body["sourceCollectionId"] = source_collection_id.strip()
            raw = await self._transport.request(
                "POST",
                library_route(scope, f"collections/{enc_col}/move-items"),
                json_body=move_body,
            )
            return raw if isinstance(raw, dict) else {"result": raw}

        raw = await self._transport.request(
            "POST",
            library_route(scope, f"collections/{enc_col}/items/bulk-detach"),
            json_body={"itemIds": clean_ids},
        )
        return raw if isinstance(raw, dict) else {"result": raw}

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
    ) -> dict[str, Any]:
        enc_id = quote(item_id.strip(), safe="")
        results: dict[str, Any] = {"item_id": item_id}

        if tags is not None:
            results["tags_update"] = await self._transport.request(
                "PATCH",
                library_route(scope, f"items/{enc_id}"),
                json_body={"tags": [str(t).strip() for t in tags if str(t).strip()]},
            )

        norm_status = normalize_read_status(read_status)
        if norm_status is not None or is_starred is not None:
            state_body: dict[str, Any] = {}
            if norm_status is not None:
                state_body["readStatus"] = norm_status
            if is_starred is not None:
                state_body["isStarred"] = is_starred
            results["state_update"] = await self._transport.request(
                "PATCH",
                library_route(scope, f"items/{enc_id}/state"),
                json_body=state_body,
            )

        clean_links = [str(i).strip() for i in link_item_ids if str(i).strip()]
        if clean_links:
            results["relations_linked"] = await self._transport.request(
                "POST",
                library_route(scope, f"items/{enc_id}/relations"),
                json_body={"targetItemIds": clean_links},
            )

        if unlink_item_id and unlink_item_id.strip():
            enc_target = quote(unlink_item_id.strip(), safe="")
            results["relation_unlinked"] = await self._transport.request(
                "DELETE",
                library_route(scope, f"items/{enc_id}/relations/{enc_target}"),
            )

        return results

    async def save_note(
        self,
        scope: LibraryScope,
        *,
        note_id: str | None = None,
        item_id: str | None = None,
        title: str | None = None,
        content_md: str = "",
        tags: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if title is not None and title.strip():
            body["title"] = title.strip()
        if content_md:
            body["contentMd"] = content_md
        if tags is not None:
            body["tags"] = [str(t).strip() for t in tags if str(t).strip()]

        if note_id and note_id.strip():
            enc_note = quote(note_id.strip(), safe="")
            raw = await self._transport.request(
                "PATCH",
                library_route(scope, f"notes/{enc_note}"),
                json_body=body,
            )
            return raw if isinstance(raw, dict) else {"result": raw}

        if item_id and item_id.strip():
            body["itemId"] = item_id.strip()
        raw = await self._transport.request("POST", library_route(scope, "notes"), json_body=body)
        return raw if isinstance(raw, dict) else {"result": raw}

    async def trash_items(
        self,
        scope: LibraryScope,
        *,
        item_ids: Sequence[str],
    ) -> dict[str, Any]:
        clean_ids = [str(i).strip() for i in item_ids if str(i).strip()]
        if not clean_ids:
            raise LibraryGatewayError(400, "trash_items requires at least one item_id")
        raw = await self._transport.request(
            "POST",
            library_route(scope, "items/bulk-trash"),
            json_body={"itemIds": clean_ids},
        )
        return raw if isinstance(raw, dict) else {"trashed_ids": clean_ids, "result": raw}

    async def merge_duplicates(
        self,
        scope: LibraryScope,
        *,
        primary_item_id: str,
        duplicate_item_ids: Sequence[str],
    ) -> dict[str, Any]:
        clean_dups = [str(i).strip() for i in duplicate_item_ids if str(i).strip()]
        if not primary_item_id.strip() or not clean_dups:
            raise LibraryGatewayError(
                400,
                "merge_duplicates requires primary_item_id and duplicate_item_ids",
            )
        raw = await self._transport.request(
            "POST",
            library_route(scope, "curation/merge"),
            json_body={
                "primaryItemId": primary_item_id.strip(),
                "duplicateItemIds": clean_dups,
            },
        )
        return raw if isinstance(raw, dict) else {"result": raw}
