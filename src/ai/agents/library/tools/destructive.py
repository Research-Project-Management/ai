"""Destructive Library tools gated behind mandatory human-in-the-loop approval."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ai.agents.library.gateway.contracts import LibraryGateway, LibraryScope
from ai.agents.library.tools.contracts import LibraryToolSpec, ToolResult


class TrashItemsArgs(BaseModel):
    item_ids: list[str] = Field(..., min_length=1, max_length=50)


class MergeDuplicatesArgs(BaseModel):
    primary_item_id: str = Field(..., min_length=1)
    duplicate_item_ids: list[str] = Field(..., min_length=1, max_length=20)


async def _handle_trash_items(
    gateway: LibraryGateway,
    scope: LibraryScope,
    args: TrashItemsArgs,
) -> ToolResult:
    data = await gateway.write.trash_items(scope, item_ids=args.item_ids)
    return ToolResult(tool="trash_items", status="ok", data=data)


async def _handle_merge_duplicates(
    gateway: LibraryGateway,
    scope: LibraryScope,
    args: MergeDuplicatesArgs,
) -> ToolResult:
    data = await gateway.write.merge_duplicates(
        scope,
        primary_item_id=args.primary_item_id,
        duplicate_item_ids=args.duplicate_item_ids,
    )
    return ToolResult(tool="merge_duplicates", status="ok", data=data)


def _label(item_id: str, known_titles: dict[str, str]) -> str:
    title = known_titles.get(item_id)
    return f'"{title}"' if title else item_id


def _describe_trash_confirmation(
    interrupt_id: str,
    validated_args: dict[str, Any],
    known_titles: dict[str, str],
) -> dict[str, Any]:
    display_args: dict[str, Any] = dict(validated_args)
    ids = [str(i) for i in validated_args.get("item_ids", ())]
    resolved_items = [{"id": i, "title": known_titles.get(i, "")} for i in ids]
    display_args["items"] = resolved_items
    summary = ", ".join(_label(i, known_titles) for i in ids[:5])
    if len(ids) > 5:
        summary += f" (+{len(ids) - 5} more)"
    noun = "item" if len(ids) == 1 else "items"
    description = f"Move {len(ids)} library {noun} to Trash: {summary}"

    return {
        "id": interrupt_id,
        "tool": "trash_items",
        "description": description,
        "args": display_args,
        "allowed_decisions": ["approve", "reject"],
    }


def _describe_merge_confirmation(
    interrupt_id: str,
    validated_args: dict[str, Any],
    known_titles: dict[str, str],
) -> dict[str, Any]:
    display_args: dict[str, Any] = dict(validated_args)
    primary_id = str(validated_args.get("primary_item_id") or "")
    dup_ids = [str(i) for i in validated_args.get("duplicate_item_ids", ())]
    display_args["primary_title"] = known_titles.get(primary_id, "")
    display_args["duplicates"] = [{"id": i, "title": known_titles.get(i, "")} for i in dup_ids]
    dup_noun = "duplicate item" if len(dup_ids) == 1 else "duplicate items"
    description = (
        f"Merge {len(dup_ids)} {dup_noun} [{', '.join(_label(i, known_titles) for i in dup_ids[:3])}] "
        f"into primary item {_label(primary_id, known_titles)}"
    )

    return {
        "id": interrupt_id,
        "tool": "merge_duplicates",
        "description": description,
        "args": display_args,
        "allowed_decisions": ["approve", "reject"],
    }


def build_destructive_tools() -> tuple[LibraryToolSpec, ...]:
    """Return destructive Library tools requiring explicit user confirmation."""
    return (
        LibraryToolSpec(
            name="trash_items",
            description=(
                "Move one or more library item_ids to Trash (soft-delete). "
                "Resolve titles via search_items first; requires user approval."
            ),
            args_schema=TrashItemsArgs,
            risk_tier="destructive",
            idempotent=False,
            handler=_handle_trash_items,
            describe_confirmation=_describe_trash_confirmation,
        ),
        LibraryToolSpec(
            name="merge_duplicates",
            description=(
                "Merge duplicate item records into primary_item_id. "
                "Resolve candidate IDs via audit_library/search_items first; requires user approval."
            ),
            args_schema=MergeDuplicatesArgs,
            risk_tier="destructive",
            idempotent=False,
            handler=_handle_merge_duplicates,
            describe_confirmation=_describe_merge_confirmation,
        ),
    )
