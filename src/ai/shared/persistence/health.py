"""Deep infrastructure health probes and concurrent evaluator."""

import asyncio
import time
from collections.abc import Sequence
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from ai.shared.config.database import DatabaseConfig
from ai.shared.contracts.health import ComponentStatus, HealthProbe
from ai.shared.persistence.redis import await_redis


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 2)


class PostgresCheckpointerProbe:
    """Health probe verifying the LangGraph checkpointer connection state."""

    name = "postgres"

    def __init__(
        self,
        checkpointer: BaseCheckpointSaver | None,
        *,
        configured: bool,
    ) -> None:
        self._checkpointer = checkpointer
        self._configured = configured

    @property
    def required(self) -> bool:
        return self._configured

    async def check(self) -> ComponentStatus:
        start = time.perf_counter()
        if not self._configured or isinstance(self._checkpointer, MemorySaver):
            return ComponentStatus(
                name=self.name,
                status="disabled",
                required=False,
                latency_ms=_elapsed_ms(start),
                detail="using in-memory checkpointer",
            )

        if self._checkpointer is None:
            return ComponentStatus(
                name=self.name,
                status="degraded",
                required=True,
                latency_ms=_elapsed_ms(start),
                detail="checkpointer unavailable",
            )

        conn = getattr(self._checkpointer, "conn", None)
        if conn is not None and not getattr(conn, "closed", False):
            return ComponentStatus(
                name=self.name,
                status="ok",
                required=True,
                latency_ms=_elapsed_ms(start),
            )

        return ComponentStatus(
            name=self.name,
            status="degraded",
            required=True,
            latency_ms=_elapsed_ms(start),
            detail="postgres connection missing or closed",
        )


class RedisProbe:
    """Health probe issuing a `PING` to Redis."""

    name = "redis"

    def __init__(self, redis_client: Redis | None, *, configured: bool) -> None:
        self._redis = redis_client
        self._configured = configured

    @property
    def required(self) -> bool:
        return self._configured

    async def check(self) -> ComponentStatus:
        start = time.perf_counter()
        if not self._configured or self._redis is None:
            return ComponentStatus(
                name=self.name,
                status="disabled",
                required=False,
                latency_ms=_elapsed_ms(start),
                detail="redis url not configured",
            )

        pong = await await_redis(self._redis.ping())
        status: Literal["ok", "degraded"] = "ok" if pong else "degraded"
        return ComponentStatus(
            name=self.name,
            status=status,
            required=True,
            latency_ms=_elapsed_ms(start),
            detail="" if pong else "ping returned falsy",
        )


class QdrantProbe:
    """Health probe issuing `get_collections` against Qdrant."""

    name = "qdrant"

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient | None,
        *,
        configured: bool,
    ) -> None:
        self._qdrant = qdrant_client
        self._configured = configured

    @property
    def required(self) -> bool:
        return self._configured

    async def check(self) -> ComponentStatus:
        start = time.perf_counter()
        if not self._configured or self._qdrant is None:
            return ComponentStatus(
                name=self.name,
                status="disabled",
                required=False,
                latency_ms=_elapsed_ms(start),
                detail="qdrant url not configured",
            )

        await self._qdrant.get_collections()
        return ComponentStatus(
            name=self.name,
            status="ok",
            required=True,
            latency_ms=_elapsed_ms(start),
        )


async def _run_probe(probe: HealthProbe, timeout_seconds: float) -> ComponentStatus:
    start = time.perf_counter()
    try:
        return await asyncio.wait_for(probe.check(), timeout=timeout_seconds)
    except TimeoutError:
        return ComponentStatus(
            name=probe.name,
            status="degraded",
            required=probe.required,
            latency_ms=_elapsed_ms(start),
            detail=f"probe timed out after {timeout_seconds}s",
        )
    except Exception as exc:  # noqa: BLE001
        return ComponentStatus(
            name=probe.name,
            status="degraded",
            required=probe.required,
            latency_ms=_elapsed_ms(start),
            detail=str(exc) or type(exc).__name__,
        )


def build_infrastructure_probes(
    database_config: DatabaseConfig,
    checkpointer: BaseCheckpointSaver | None,
    redis_client: Redis | None,
    qdrant_client: AsyncQdrantClient | None,
) -> tuple[HealthProbe, ...]:
    """Assemble the standard PostgreSQL, Redis, and Qdrant health probes."""
    return (
        PostgresCheckpointerProbe(
            checkpointer,
            configured=database_config.postgres.is_configured,
        ),
        RedisProbe(
            redis_client,
            configured=database_config.redis.is_configured,
        ),
        QdrantProbe(
            qdrant_client,
            configured=database_config.qdrant.is_configured,
        ),
    )


async def evaluate_health(
    probes: Sequence[HealthProbe],
    *,
    timeout_seconds: float = 2.0,
) -> tuple[Literal["ok", "degraded"], list[ComponentStatus]]:
    """Run all probes concurrently and derive overall readiness status."""
    if not probes:
        return "ok", []

    results = list(
        await asyncio.gather(*(_run_probe(probe, timeout_seconds) for probe in probes))
    )
    overall: Literal["ok", "degraded"] = (
        "degraded"
        if any(item.required and item.status == "degraded" for item in results)
        else "ok"
    )
    return overall, results
