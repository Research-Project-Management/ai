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


__all__ = [
    "as_dict",
    "encode_seg",
    "unwrap_nested_dict",
]
