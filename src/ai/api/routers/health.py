"""Deep readiness and resilience observability endpoints."""

import inspect
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, Response, status

from ai.api.dependencies import get_health_probes, get_server_config, verify_internal_key
from ai.api.schemas import HealthResponse
from ai.shared.config.server import ServerConfig
from ai.shared.contracts.health import HealthProbe
from ai.shared.persistence.health import evaluate_health
from ai.shared.resilience.prometheus import render_prometheus_metrics

router = APIRouter(tags=["health"])


def _collect_runtime_snapshots(
    state: Any,
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any], dict[str, int]]:
    breakers = getattr(state, "circuit_breakers", None)
    tool_metrics = getattr(state, "tool_metrics", None)
    llm_metrics = getattr(state, "llm_metrics", None)
    ingestion_runner = getattr(state, "ingestion_job_runner", None)
    return (
        breakers.snapshot() if breakers is not None else {},
        tool_metrics.snapshot() if tool_metrics is not None else {},
        llm_metrics.snapshot() if llm_metrics is not None else {},
        ingestion_runner.snapshot() if ingestion_runner is not None else {},
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
async def resilience_health(request: Request) -> dict[str, Any]:
    """Return Circuit Breaker states, tool & LLM metrics, queue depth, token quotas & provenance."""
    state = request.app.state
    breakers, tool_metrics, llm_metrics, ingestion_queue = _collect_runtime_snapshots(state)
    budget = getattr(state, "token_budget", None)
    facade = getattr(state, "context_facade", None)

    token_budget_snapshot: dict[str, Any] = {}
    if budget is not None:
        raw_snapshot = budget.snapshot()
        token_budget_snapshot = (
            await raw_snapshot if inspect.isawaitable(raw_snapshot) else raw_snapshot
        )

    return {
        "circuit_breakers": breakers,
        "tool_metrics": tool_metrics,
        "llm_metrics": llm_metrics,
        "ingestion_queue": ingestion_queue,
        "token_budget": token_budget_snapshot,
        "provenance": facade.get_summary_metrics() if facade is not None else {},
    }


@router.get("/metrics", dependencies=[Depends(verify_internal_key)])
async def prometheus_metrics(request: Request) -> Response:
    """Expose runtime tool, LLM, breaker, and ingestion queue metrics in Prometheus 0.0.4 text."""
    breakers, tool_metrics, llm_metrics, ingestion_queue = _collect_runtime_snapshots(
        request.app.state
    )
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
