"""Shared JSON parsing and URL-encoding helpers for Project REST sub-clients."""

from __future__ import annotations

from typing import Any

from ai.shared.gateway.transport import encode_seg


def as_dict(value: Any) -> dict[str, Any]:
    """Narrow dynamic JSON values to `dict[str, Any]` for strict static typing."""
    return value if isinstance(value, dict) else {}


def unwrap_nested_dict(
    raw: Any,
    key: str,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract an entity dict from `{key: {...}}` or a flat `{...}` response payload."""
    if isinstance(raw, dict):
        nested = raw.get(key)
        if isinstance(nested, dict):
            return nested
        return raw
    return fallback if fallback is not None else {}


def work_item_route(work_item_id: str) -> str:
    """Return the REST path for a work item resource."""
    return f"/api/v1/work-items/{encode_seg(work_item_id)}"


def work_item_comment_route(comment_id: str) -> str:
    """Return the REST path for a work item comment resource."""
    return f"/api/v1/work-items/comments/{encode_seg(comment_id)}"


def project_label_route(project_id: str, label_id: str) -> str:
    """Return the REST path for a project label resource."""
    return f"/api/v1/projects/{encode_seg(project_id)}/labels/{encode_seg(label_id)}"


__all__ = [
    "as_dict",
    "encode_seg",
    "project_label_route",
    "unwrap_nested_dict",
    "work_item_comment_route",
    "work_item_route",
]
