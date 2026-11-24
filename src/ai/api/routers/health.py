"""Deep readiness and resilience observability endpoints."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response, status

from ai.api.dependencies import (
    get_container,
    get_health_probes,
    get_server_config,
    get_token_budget,
    verify_internal_key,
)
from ai.api.schemas import HealthResponse
from ai.bootstrap import AppContainer
from ai.shared.config.server import ServerConfig
from ai.shared.contracts.health import HealthProbe
from ai.shared.persistence.health import evaluate_health
from ai.shared.resilience.observability import TokenBudget
from ai.shared.resilience.prometheus import render_prometheus_metrics

router = APIRouter(tags=["health"])


def _collect_runtime_snapshots(
    container: AppContainer,
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any], dict[str, int]]:
    return (
        container.circuit_breakers.snapshot(),
        container.tool_metrics.snapshot(),
        container.llm_metrics.snapshot(),
        container.ingestion_job_runner.snapshot(),
    )


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
async def resilience_health(
    container: Annotated[AppContainer, Depends(get_container)],
    token_budget: Annotated[TokenBudget, Depends(get_token_budget)],
) -> dict[str, Any]:
    """Return Circuit Breaker states, tool & LLM metrics, queue depth, and token quotas."""
    breakers, tool_metrics, llm_metrics, ingestion_queue = _collect_runtime_snapshots(container)
    return {
        "circuit_breakers": breakers,
        "tool_metrics": tool_metrics,
        "llm_metrics": llm_metrics,
        "ingestion_queue": ingestion_queue,
        "token_budget": await token_budget.snapshot(),
    }


@router.get("/metrics", dependencies=[Depends(verify_internal_key)])
async def prometheus_metrics(
    container: Annotated[AppContainer, Depends(get_container)],
) -> Response:
    """Expose runtime tool, LLM, breaker, and ingestion queue metrics in Prometheus 0.0.4 text."""
    breakers, tool_metrics, llm_metrics, ingestion_queue = _collect_runtime_snapshots(container)
    payload = render_prometheus_metrics(
        tool_metrics=tool_metrics,
        llm_metrics=llm_metrics,
        circuit_breakers=breakers,
        ingestion_queue=ingestion_queue,
    )
    return Response(
        content=payload,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
