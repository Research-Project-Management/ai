"""Zero-dependency Prometheus text exposition (v0.0.4) formatter for AI runtime metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

_BREAKER_STATE_VALUES: dict[str, int] = {
    "closed": 0,
    "half_open": 1,
    "open": 2,
}


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_labels(labels: Mapping[str, str]) -> str:
    if not labels:
        return ""
    pairs = [f'{k}="{_escape_label(v)}"' for k, v in labels.items()]
    return "{" + ",".join(pairs) + "}"


def _append_metric_family(
    lines: list[str],
    name: str,
    metric_type: Literal["counter", "gauge"],
    help_text: str,
    samples: Sequence[tuple[dict[str, str], int | float]],
) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {metric_type}")
    for labels, value in samples:
        lines.append(f"{name}{_format_labels(labels)} {value}")


def render_prometheus_metrics(
    *,
    tool_metrics: Mapping[str, Mapping[str, Any]],
    llm_metrics: Mapping[str, Mapping[str, Any]],
    circuit_breakers: Mapping[str, str],
    ingestion_queue: Mapping[str, int],
) -> str:
    """Render tool, LLM, circuit breaker, and ingestion queue stats in Prometheus 0.0.4 format."""
    lines: list[str] = []

    tool_calls: list[tuple[dict[str, str], int | float]] = []
    tool_errors: list[tuple[dict[str, str], int | float]] = []
    tool_cache_hits: list[tuple[dict[str, str], int | float]] = []
    tool_latency: list[tuple[dict[str, str], int | float]] = []
    for tool_name, stat in sorted(tool_metrics.items()):
        lbl = {"tool": str(tool_name)}
        tool_calls.append((lbl, int(stat.get("calls", 0))))
        tool_errors.append((lbl, int(stat.get("errors", 0))))
        tool_cache_hits.append((lbl, int(stat.get("cache_hits", 0))))
        tool_latency.append((lbl, float(stat.get("total_latency_ms", 0.0))))

    _append_metric_family(
        lines, "ai_tool_calls_total", "counter", "Total AI tool calls", tool_calls
    )
    _append_metric_family(
        lines, "ai_tool_errors_total", "counter", "Total failed AI tool calls", tool_errors
    )
    _append_metric_family(
        lines,
        "ai_tool_cache_hits_total",
        "counter",
        "Total AI tool cache hits",
        tool_cache_hits,
    )
    _append_metric_family(
        lines,
        "ai_tool_latency_ms_sum",
        "counter",
        "Cumulative AI tool execution latency in milliseconds",
        tool_latency,
    )

    llm_calls: list[tuple[dict[str, str], int | float]] = []
    llm_errors: list[tuple[dict[str, str], int | float]] = []
    llm_tokens: list[tuple[dict[str, str], int | float]] = []
    llm_latency: list[tuple[dict[str, str], int | float]] = []
    for _key, stat in sorted(llm_metrics.items()):
        role = str(stat.get("role", "default"))
        model = str(stat.get("model", "unknown"))
        base_lbl = {"role": role, "model": model}
        llm_calls.append((base_lbl, int(stat.get("calls", 0))))
        llm_errors.append((base_lbl, int(stat.get("errors", 0))))
        llm_tokens.append(({**base_lbl, "direction": "in"}, int(stat.get("input_tokens", 0))))
        llm_tokens.append(({**base_lbl, "direction": "out"}, int(stat.get("output_tokens", 0))))
        llm_latency.append((base_lbl, float(stat.get("total_latency_ms", 0.0))))

    _append_metric_family(lines, "ai_llm_calls_total", "counter", "Total AI LLM calls", llm_calls)
    _append_metric_family(
        lines, "ai_llm_errors_total", "counter", "Total failed AI LLM calls", llm_errors
    )
    _append_metric_family(
        lines,
        "ai_llm_tokens_total",
        "counter",
        "Total AI LLM tokens processed by direction",
        llm_tokens,
    )
    _append_metric_family(
        lines,
        "ai_llm_latency_ms_sum",
        "counter",
        "Cumulative AI LLM call latency in milliseconds",
        llm_latency,
    )

    breaker_samples: list[tuple[dict[str, str], int | float]] = [
        ({"service": str(service)}, _BREAKER_STATE_VALUES.get(str(state).lower(), 0))
        for service, state in sorted(circuit_breakers.items())
    ]
    _append_metric_family(
        lines,
        "ai_circuit_breaker_state",
        "gauge",
        "Circuit breaker state (0=closed, 1=half_open, 2=open)",
        breaker_samples,
    )

    active_jobs = int(ingestion_queue.get("active_jobs", 0))
    max_queue_size = int(ingestion_queue.get("max_queue_size", 0))
    _append_metric_family(
        lines,
        "ai_ingestion_queue_active",
        "gauge",
        "Current number of queued or running ingestion jobs",
        [({}, active_jobs)],
    )
    _append_metric_family(
        lines,
        "ai_ingestion_queue_capacity",
        "gauge",
        "Maximum configured ingestion queue capacity",
        [({}, max_queue_size)],
    )

    return "\n".join(lines) + "\n"
