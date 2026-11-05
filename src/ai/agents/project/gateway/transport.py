"""Project Backend transport: shared `BackendTransport` bound to `ProjectGatewayError`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx

from ai.agents.project.gateway.contracts import ProjectGatewayError
from ai.shared.contracts.auth import get_delegation_token
from ai.shared.gateway.transport import BackendTransport, encode_seg


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


class ProjectHttpTransport(BackendTransport):
    """Shared async HTTP transport for the Project REST sub-clients."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 15.0,
        http_client: httpx.AsyncClient | None = None,
        token_provider: Callable[[], str] = get_delegation_token,
        internal_api_key: str = "",
    ) -> None:
        super().__init__(
            base_url=base_url,
            service="Project",
            error_factory=ProjectGatewayError,
            timeout_seconds=timeout_seconds,
            http_client=http_client,
            token_provider=token_provider,
            internal_api_key=internal_api_key,
        )


__all__ = [
    "ProjectHttpTransport",
    "as_dict",
    "encode_seg",
    "unwrap_nested_dict",
]
