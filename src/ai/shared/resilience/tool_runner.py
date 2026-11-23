"""Resilient execution helper (Timeout, Bounded Retry, Circuit Breaker, TTL Cache, Metrics)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, cast

from ai.shared.resilience.circuit_breaker import CircuitBreaker
from ai.shared.resilience.contracts import ToolCache
from ai.shared.resilience.observability import ToolExecutionMetrics

_ACTIVE_METRICS: ContextVar[ToolExecutionMetrics | None] = ContextVar(
    "flux_tool_metrics", default=None
)


@contextmanager
def bind_tool_metrics(metrics: ToolExecutionMetrics | None) -> Iterator[None]:
    """Record every tool call made inside this context into `metrics`."""
    token = _ACTIVE_METRICS.set(metrics)
    try:
        yield
    finally:
        _ACTIVE_METRICS.reset(token)


def _record(name: str, started: float, *, ok: bool, cache_hit: bool = False) -> None:
    metrics = _ACTIVE_METRICS.get()
    if metrics is not None:
        metrics.record_call(
            name,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            ok=ok,
            cache_hit=cache_hit,
        )


async def execute_tool_safely[T](
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    timeout_seconds: float = 15.0,
    max_retries: int = 1,
    circuit_breaker: CircuitBreaker | None = None,
    cache: ToolCache | None = None,
    cache_key: str | None = None,
    cache_ttl_seconds: int = 300,
    tool_name: str | None = None,
    **kwargs: Any,
) -> T:
    """Invoke an async tool guarded by optional TTL cache, Circuit Breaker, timeout & retries."""
    name = tool_name or getattr(fn, "__name__", "tool")
    started = time.perf_counter()
    if cache is not None and cache_key:
        cached = await cache.get(cache_key)
        if cached is not None:
            _record(name, started, ok=True, cache_hit=True)
            return cast(T, cached)

    if circuit_breaker is not None:
        await circuit_breaker.allow_request()

    attempts = max(0, max_retries) + 1
    last_exc: Exception | None = None
    try:
        for _ in range(attempts):
            try:
                result = await asyncio.wait_for(fn(*args, **kwargs), timeout=timeout_seconds)
                if circuit_breaker is not None:
                    await circuit_breaker.record_success()
                if cache is not None and cache_key:
                    await cache.set(cache_key, result, ttl_seconds=cache_ttl_seconds)
                _record(name, started, ok=True)
                return result
            except (TimeoutError, OSError) as exc:
                last_exc = exc
        assert last_exc is not None
        raise last_exc
    except Exception:
        if circuit_breaker is not None:
            await circuit_breaker.record_failure()
        _record(name, started, ok=False)
        raise
