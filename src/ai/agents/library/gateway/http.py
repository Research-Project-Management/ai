"""HTTP `LibraryGateway`: one shared transport feeding the read and write clients."""

from __future__ import annotations

from collections.abc import Callable

import httpx

from ai.agents.library.gateway.contracts import LibraryGatewayError
from ai.agents.library.gateway.reader import LibraryReader
from ai.agents.library.gateway.writer import LibraryWriter
from ai.shared.contracts.auth import get_delegation_token
from ai.shared.gateway.transport import BackendTransport


class HttpLibraryGateway:
    """Flux Backend Library APIs exposed as `read` and `write` ports with user JWT delegation."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 15.0,
        max_items_per_call: int = 20,
        http_client: httpx.AsyncClient | None = None,
        token_provider: Callable[[], str] = get_delegation_token,
    ) -> None:
        self.transport = BackendTransport(
            base_url=base_url,
            service="Library",
            error_factory=LibraryGatewayError,
            timeout_seconds=timeout_seconds,
            http_client=http_client,
            token_provider=token_provider,
        )
        self.read = LibraryReader(self.transport, max_items_per_call=max_items_per_call)
        self.write = LibraryWriter(self.transport)
