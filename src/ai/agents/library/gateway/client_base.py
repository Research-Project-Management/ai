"""Route selection helper for Library REST clients."""

from __future__ import annotations

from typing import Final
from urllib.parse import quote

from ai.agents.library.gateway.contracts import LibraryScope

_PERSONAL_SENTINELS: Final[frozenset[str]] = frozenset({"personal", "me", "all", "user", "flux"})


def _is_personal_scope(scope: LibraryScope) -> bool:
    if scope.scope_type == "personal":
        return True
    if scope.scope_type == "project":
        return False
    space = scope.space_id.strip()
    if not space or space.lower() in _PERSONAL_SENTINELS:
        return True
    user_id = scope.user_id.strip()
    return bool(user_id and space == user_id)


def library_route(scope: LibraryScope, suffix: str) -> str:
    """Build personal or space-scoped Library API path."""
    clean_suffix = suffix.lstrip("/")
    if _is_personal_scope(scope):
        return f"/api/v1/library/{clean_suffix}"
    encoded_space = quote(scope.space_id.strip(), safe="")
    return f"/api/v1/spaces/{encoded_space}/library/{clean_suffix}"
