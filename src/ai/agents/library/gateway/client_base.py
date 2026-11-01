"""Shared plumbing for Library REST clients: route selection and transport access."""

from __future__ import annotations

from typing import Any, Final
from urllib.parse import quote

from ai.agents.library.gateway.contracts import LibraryScope
from ai.shared.gateway.transport import BackendTransport

_PERSONAL_SENTINELS: Final[frozenset[str]] = frozenset(
    {"personal", "me", "all", "user", "flux"}
)


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


class LibraryClientBase:
    """Base for the read and write clients: owns the transport and personal/space routing."""

    def __init__(self, transport: BackendTransport) -> None:
        self._transport = transport

    def _route(self, scope: LibraryScope, suffix: str) -> str:
        clean_suffix = suffix.lstrip("/")
        if _is_personal_scope(scope):
            return f"/api/v1/library/{clean_suffix}"
        encoded_space = quote(scope.space_id.strip(), safe="")
        return f"/api/v1/spaces/{encoded_space}/library/{clean_suffix}"

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        return await self._transport.request(method, path, params=params, json_body=json_body)
