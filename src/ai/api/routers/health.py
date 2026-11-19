"""Deep readiness and resilience observability endpoints."""

import inspect
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, Response, status

from ai.api.dependencies import get_health_probes, get_server_config, verify_internal_key
from ai.api.schemas import HealthResponse
from ai.shared.config.server import ServerConfig
from ai.shared.contracts.health import HealthProbe
from ai.shared.persistence.health import evaluate_health

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(
    response: Response,
    server_cfg: Annotated[ServerConfig, Depends(get_server_config)],
    probes: Annotated[tuple[HealthProbe, ...], Depends(get_health_probes)],
) -> HealthResponse:
    """Run Postgres, Redis, and Qdrant probes concurrently and report 503 if degraded."""
    overall_status, components = await evaluate_health(probes, timeout_seconds=2.0)
    if overall_status == "degraded":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status=overall_status,
        app_env=server_cfg.app_env.value,
        components=components,
    )


@router.get("/health/resilience", dependencies=[Depends(verify_internal_key)])
async def resilience_health(request: Request) -> dict[str, Any]:
    """Return Circuit Breaker states, tool latency metrics, token quotas & provenance stats."""
    state = request.app.state
    breakers = getattr(state, "circuit_breakers", None)
    metrics = getattr(state, "tool_metrics", None)
    budget = getattr(state, "token_budget", None)
    facade = getattr(state, "context_facade", None)

    token_budget_snapshot: dict[str, Any] = {}
    if budget is not None:
        raw_snapshot = budget.snapshot()
        token_budget_snapshot = (
            await raw_snapshot
            if inspect.isawaitable(raw_snapshot)
            else raw_snapshot
        )

    return {
        "circuit_breakers": breakers.snapshot() if breakers is not None else {},
        "tool_metrics": metrics.snapshot() if metrics is not None else {},
        "token_budget": token_budget_snapshot,
        "provenance": facade.get_summary_metrics() if facade is not None else {},
    }
