"""Shared execution of one approved tool batch (the `execute` node body of every tool agent)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp
from pydantic import BaseModel

from ai.shared.resilience.tool_runner import execute_tool_safely
from ai.shared.workflow.batch import (
    BatchTracker,
    HasProvenance,
    dedupe_citations,
    observation_of,
)
from ai.shared.workflow.events import ToolEventPublisher, summarize_tool_input

logger = logging.getLogger(__name__)


class RunnableSpec(Protocol):
    """Structural view of a tool spec needed to execute it."""

    @property
    def name(self) -> str: ...
    @property
    def risk_tier(self) -> str: ...
    @property
    def idempotent(self) -> bool: ...
    @property
    def args_schema(self) -> type[BaseModel]: ...


class ToolOutcome(Protocol):
    """Structural view of a tool result (satisfied by every agent's result model)."""

    @property
    def tool(self) -> str: ...
    @property
    def status(self) -> str: ...
    @property
    def data(self) -> Any: ...
    @property
    def error(self) -> str | None: ...
    @property
    def citations(self) -> Sequence[Any]: ...


@dataclass(frozen=True)
class BatchOutcome[C: HasProvenance]:
    """Observations and citations after the batch, plus whether the turn can end now."""

    observations: list[dict[str, Any]]
    citations: list[C]
    is_terminal: bool


async def run_tool_batch[R: ToolOutcome, C: HasProvenance](
    *,
    approved: Sequence[Mapping[str, Any]],
    get_spec: Callable[[str], Any],
    invoke: Callable[[Any, BaseModel], Awaitable[R]],
    failure: Callable[[str, str], R],
    gateway_error: type[Exception],
    events: ToolEventPublisher,
    config: RunnableConfig,
    timeout_seconds: float,
    observations: Sequence[dict[str, Any]],
    citations: Sequence[C],
    after_result: Callable[[R], Awaitable[None]] | None = None,
) -> BatchOutcome[C]:
    """Run approved calls in order with timeout/retry, failure isolation, and live events.

    `invoke(spec, args)` performs the agent-specific call. Failures become `failure(tool, msg)`
    results so one bad tool never aborts the batch; `GraphBubbleUp` (HITL) always propagates.
    """
    obs = list(observations)
    cites = list(citations)
    batch = BatchTracker(expected_calls=len(approved))

    async def _run_one(
        spec: RunnableSpec,
        args_mapping: Mapping[str, Any],
    ) -> R:
        args = spec.args_schema.model_validate(args_mapping)
        await events.emit_tool_started(
            spec.name,
            config,
            tool_input=summarize_tool_input(args_mapping) or None,
        )
        try:
            result = await execute_tool_safely(
                invoke,
                spec,
                args,
                timeout_seconds=timeout_seconds,
                max_retries=1 if spec.idempotent else 0,
                tool_name=spec.name,
            )
        except GraphBubbleUp:
            raise
        except gateway_error as exc:
            result = failure(spec.name, str(getattr(exc, "detail", exc)))
        except Exception as exc:
            logger.warning("Tool '%s' failed: %s", spec.name, exc, exc_info=True)
            result = failure(spec.name, f"{type(exc).__name__}: {exc}")
        return result

    async def _record_one(spec: RunnableSpec, result: R) -> None:
        nonlocal cites
        batch.observe(risk_tier=spec.risk_tier, status=result.status)
        if after_result is not None:
            await after_result(result)
        await events.emit_tool_completed(spec.name, result.status, config, error=result.error)
        obs.append(observation_of(result))
        cites = dedupe_citations(cites, list(result.citations))

    idx = 0
    total = len(approved)
    while idx < total:
        call = approved[idx]
        spec = get_spec(str(call.get("tool") or ""))
        if spec is None:
            batch.observe(risk_tier=None, status=None)
            idx += 1
            continue

        if spec.risk_tier == "read":
            read_group: list[tuple[RunnableSpec, Mapping[str, Any]]] = []
            while idx < total:
                next_call = approved[idx]
                next_spec = get_spec(str(next_call.get("tool") or ""))
                if next_spec is None or next_spec.risk_tier != "read":
                    break
                raw_args = next_call.get("args")
                read_group.append((next_spec, raw_args if isinstance(raw_args, Mapping) else {}))
                idx += 1
            if len(read_group) == 1:
                single_spec, single_args = read_group[0]
                await _record_one(single_spec, await _run_one(single_spec, single_args))
            else:
                results = await asyncio.gather(
                    *(_run_one(s, a) for s, a in read_group),
                )
                for (g_spec, _), res in zip(read_group, results, strict=True):
                    await _record_one(g_spec, res)
            continue

        raw_args = call.get("args")
        args_mapping = raw_args if isinstance(raw_args, Mapping) else {}
        await _record_one(spec, await _run_one(spec, args_mapping))
        idx += 1

    return BatchOutcome(observations=obs, citations=cites, is_terminal=batch.is_terminal)
