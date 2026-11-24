"""HTTP `ProjectGateway`: one shared transport feeding three area clients."""

from __future__ import annotations

from collections.abc import Callable

import httpx

from ai.agents.project.gateway.analytics_client import AnalyticsClient
from ai.agents.project.gateway.contracts import ProjectGatewayError
from ai.agents.project.gateway.projects_client import ProjectsClient
from ai.agents.project.gateway.work_items_client import WorkItemsClient
from ai.shared.contracts.auth import get_delegation_token
from ai.shared.gateway.transport import BackendTransport


class HttpProjectGateway:
    """Flux NestJS Project APIs exposed as `projects`, `work_items`, and `analytics` ports."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 15.0,
        http_client: httpx.AsyncClient | None = None,
        token_provider: Callable[[], str] = get_delegation_token,
        internal_api_key: str = "",
    ) -> None:
        transport = BackendTransport(
            base_url=base_url,
            service="Project",
            error_factory=ProjectGatewayError,
            timeout_seconds=timeout_seconds,
            http_client=http_client,
            token_provider=token_provider,
            internal_api_key=internal_api_key,
        )
        self.projects = ProjectsClient(transport)
        self.work_items = WorkItemsClient(transport)
        self.analytics = AnalyticsClient(transport)
