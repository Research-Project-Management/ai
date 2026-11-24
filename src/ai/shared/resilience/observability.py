"""Tool execution metrics and per-space token budgets (in-memory and Redis)."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field
from redis.asyncio import Redis
from redis.exceptions import RedisError

from ai.shared.contracts.run_control import RunUsage
from ai.shared.persistence.redis import await_redis

logger = logging.getLogger(__name__)


class TokenQuotaStatus(BaseModel):
    """Immutable token budget and estimated USD spend snapshot for one scope."""

    model_config = ConfigDict(frozen=True)

    allowed: bool
    used_tokens: int = Field(ge=0)
    limit_tokens: int = Field(ge=0)
    remaining_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0.0)


@runtime_checkable
class TokenBudget(Protocol):
    """Port defining async token quota checks and consumption accounting."""

    async def check(self, scope: str) -> TokenQuotaStatus: ...

    async def record(self, scope: str, usage: RunUsage) -> TokenQuotaStatus: ...

    async def snapshot(self) -> dict[str, Any]: ...


@dataclass
class _ToolStat:
    calls: int = 0
    errors: int = 0
    cache_hits: int = 0
    total_latency_ms: float = 0.0


class ToolExecutionMetrics:
    """Thread-safe per-tool latency, error, and cache-hit counter registry."""

    def __init__(self) -> None:
        self._stats: dict[str, _ToolStat] = defaultdict(_ToolStat)
        self._lock = threading.Lock()

    def record_call(
        self,
        tool_name: str,
        *,
        latency_ms: float,
        ok: bool,
        cache_hit: bool = False,
    ) -> None:
        """Record a single tool execution outcome and elapsed latency in milliseconds."""
        clean_name = tool_name.strip() or "unknown"
        bounded_latency = max(0.0, float(latency_ms))
        with self._lock:
            stat = self._stats[clean_name]
            stat.calls += 1
            if not ok:
                stat.errors += 1
            if cache_hit:
                stat.cache_hits += 1
            stat.total_latency_ms += bounded_latency

    def snapshot(self) -> dict[str, dict[str, int | float]]:
        """Return a JSON-ready metrics breakdown per tool including computed avg_latency_ms."""
        with self._lock:
            out: dict[str, dict[str, int | float]] = {}
            for name, stat in self._stats.items():
                avg = stat.total_latency_ms / stat.calls if stat.calls > 0 else 0.0
                out[name] = {
                    "calls": stat.calls,
                    "errors": stat.errors,
                    "cache_hits": stat.cache_hits,
                    "total_latency_ms": round(stat.total_latency_ms, 2),
                    "avg_latency_ms": round(avg, 2),
                }
            return out


@dataclass
class _LLMStat:
    role: str
    model: str
    calls: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_latency_ms: float = 0.0


class LLMCallMetrics:
    """Thread-safe per-(role, model) LLM latency, error, and token counter registry."""

    def __init__(self) -> None:
        self._stats: dict[str, _LLMStat] = {}
        self._lock = threading.Lock()

    def record_call(
        self,
        role: str,
        model: str,
        *,
        latency_ms: float,
        ok: bool,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record a single LLM invocation by `(role, model)` with latency and token usage."""
        clean_role = role.strip() or "default"
        clean_model = model.strip() or "unknown"
        key = f"{clean_role}:{clean_model}"
        bounded_latency = max(0.0, float(latency_ms))
        in_tok = max(0, int(input_tokens))
        out_tok = max(0, int(output_tokens))
        with self._lock:
            stat = self._stats.get(key)
            if stat is None:
                stat = _LLMStat(role=clean_role, model=clean_model)
                self._stats[key] = stat
            stat.calls += 1
            if not ok:
                stat.errors += 1
            stat.input_tokens += in_tok
            stat.output_tokens += out_tok
            stat.total_latency_ms += bounded_latency

    def snapshot(self) -> dict[str, dict[str, int | float | str]]:
        """Return JSON-ready breakdown keyed by `role:model` including `avg_latency_ms`."""
        with self._lock:
            out: dict[str, dict[str, int | float | str]] = {}
            for key, stat in self._stats.items():
                avg = stat.total_latency_ms / stat.calls if stat.calls > 0 else 0.0
                out[key] = {
                    "role": stat.role,
                    "model": stat.model,
                    "calls": stat.calls,
                    "errors": stat.errors,
                    "input_tokens": stat.input_tokens,
                    "output_tokens": stat.output_tokens,
                    "total_latency_ms": round(stat.total_latency_ms, 2),
                    "avg_latency_ms": round(avg, 2),
                }
            return out


_DEFAULT_MAX_TOKENS_PER_WINDOW = 500_000
_DEFAULT_WINDOW_SECONDS = 86_400
_DEFAULT_COST_PER_1K_INPUT_USD = 0.0005
_DEFAULT_COST_PER_1K_OUTPUT_USD = 0.0015


def _compute_usage_cost(
    usage: RunUsage,
    cost_in_per_1k: float,
    cost_out_per_1k: float,
) -> tuple[int, float]:
    """Compute effective total tokens and estimated USD cost for one `RunUsage`."""
    tokens = max(usage.total_tokens, usage.input_tokens + usage.output_tokens)
    cost_usd = (usage.input_tokens / 1000.0) * cost_in_per_1k + (
        usage.output_tokens / 1000.0
    ) * cost_out_per_1k
    return tokens, cost_usd


@dataclass(frozen=True)
class _UsageEvent:
    timestamp: float
    total_tokens: int
    cost_usd: float


class InMemoryTokenBudget:
    """Sliding-window token quota and estimated cost per scope (single-process adapter)."""

    def __init__(
        self,
        *,
        max_tokens_per_window: int = _DEFAULT_MAX_TOKENS_PER_WINDOW,
        window_seconds: int = _DEFAULT_WINDOW_SECONDS,
        cost_per_1k_input_usd: float = _DEFAULT_COST_PER_1K_INPUT_USD,
        cost_per_1k_output_usd: float = _DEFAULT_COST_PER_1K_OUTPUT_USD,
    ) -> None:
        self._max_tokens = max(0, max_tokens_per_window)
        self._window_seconds = max(1, window_seconds)
        self._cost_in_per_1k = max(0.0, cost_per_1k_input_usd)
        self._cost_out_per_1k = max(0.0, cost_per_1k_output_usd)
        self._events: dict[str, deque[_UsageEvent]] = defaultdict(deque)

    def _bucket(self, scope: str, now: float) -> deque[_UsageEvent]:
        bucket = self._events[scope.strip() or "default"]
        cutoff = now - self._window_seconds
        while bucket and bucket[0].timestamp <= cutoff:
            bucket.popleft()
        return bucket

    def _status(self, bucket: deque[_UsageEvent]) -> TokenQuotaStatus:
        used = sum(ev.total_tokens for ev in bucket)
        return TokenQuotaStatus(
            allowed=used < self._max_tokens,
            used_tokens=used,
            limit_tokens=self._max_tokens,
            remaining_tokens=max(0, self._max_tokens - used),
            estimated_cost_usd=round(sum(ev.cost_usd for ev in bucket), 6),
        )

    async def check(self, scope: str) -> TokenQuotaStatus:
        return self._status(self._bucket(scope, time.monotonic()))

    async def record(self, scope: str, usage: RunUsage) -> TokenQuotaStatus:
        now = time.monotonic()
        bucket = self._bucket(scope, now)
        tokens, cost_usd = _compute_usage_cost(usage, self._cost_in_per_1k, self._cost_out_per_1k)
        if tokens > 0 or cost_usd > 0.0:
            bucket.append(_UsageEvent(timestamp=now, total_tokens=tokens, cost_usd=cost_usd))
        return self._status(bucket)

    async def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        active_scopes = 0
        total_tokens = 0
        total_cost = 0.0
        for scope in list(self._events):
            bucket = self._bucket(scope, now)
            if not bucket:
                del self._events[scope]
                continue
            active_scopes += 1
            total_tokens += sum(ev.total_tokens for ev in bucket)
            total_cost += sum(ev.cost_usd for ev in bucket)
        return {
            "active_scopes": active_scopes,
            "total_tokens": total_tokens,
            "total_estimated_cost_usd": round(total_cost, 6),
            "limit_tokens_per_window": self._max_tokens,
        }


class RedisTokenBudget:
    """Fail-open distributed Redis adapter implementing `TokenBudget` across workers."""

    def __init__(
        self,
        redis: Redis,
        *,
        max_tokens_per_window: int = _DEFAULT_MAX_TOKENS_PER_WINDOW,
        window_seconds: int = _DEFAULT_WINDOW_SECONDS,
        cost_per_1k_input_usd: float = _DEFAULT_COST_PER_1K_INPUT_USD,
        cost_per_1k_output_usd: float = _DEFAULT_COST_PER_1K_OUTPUT_USD,
        key_prefix: str = "flux:budget:",
    ) -> None:
        self._redis = redis
        self._max_tokens = max(0, max_tokens_per_window)
        self._window_seconds = max(1, window_seconds)
        self._cost_in_per_1k = max(0.0, cost_per_1k_input_usd)
        self._cost_out_per_1k = max(0.0, cost_per_1k_output_usd)
        self._prefix = key_prefix
        self._scopes_key = f"{self._prefix}scopes"

    def _tokens_key(self, scope: str) -> str:
        return f"{self._prefix}tokens:{scope}"

    def _cost_key(self, scope: str) -> str:
        return f"{self._prefix}cost:{scope}"

    def _status(self, used_tokens: int, cost_micros: int) -> TokenQuotaStatus:
        remaining = max(0, self._max_tokens - used_tokens)
        return TokenQuotaStatus(
            allowed=(used_tokens < self._max_tokens),
            used_tokens=used_tokens,
            limit_tokens=self._max_tokens,
            remaining_tokens=remaining,
            estimated_cost_usd=round(cost_micros / 1_000_000.0, 6),
        )

    def _fail_open_status(self) -> TokenQuotaStatus:
        return TokenQuotaStatus(
            allowed=True,
            used_tokens=0,
            limit_tokens=self._max_tokens,
            remaining_tokens=self._max_tokens,
            estimated_cost_usd=0.0,
        )

    async def check(self, scope: str) -> TokenQuotaStatus:
        clean_scope = scope.strip() or "default"
        try:
            tok_key = self._tokens_key(clean_scope)
            cost_key = self._cost_key(clean_scope)
            async with self._redis.pipeline(transaction=False) as pipe:
                pipe.get(tok_key)
                pipe.get(cost_key)
                raw_tok, raw_cost = await await_redis(pipe.execute())
            used = int(raw_tok) if raw_tok is not None else 0
            cost_micros = int(raw_cost) if raw_cost is not None else 0
            return self._status(used, cost_micros)
        except (RedisError, OSError, TimeoutError, ValueError) as exc:
            logger.warning("RedisTokenBudget.check failed (fail-open) for %s: %s", clean_scope, exc)
            return self._fail_open_status()

    async def record(self, scope: str, usage: RunUsage) -> TokenQuotaStatus:
        clean_scope = scope.strip() or "default"
        tokens, cost_usd = _compute_usage_cost(usage, self._cost_in_per_1k, self._cost_out_per_1k)
        cost_micros = max(0, round(cost_usd * 1_000_000))
        if tokens <= 0 and cost_micros <= 0:
            return await self.check(clean_scope)

        try:
            tok_key = self._tokens_key(clean_scope)
            cost_key = self._cost_key(clean_scope)
            async with self._redis.pipeline(transaction=False) as pipe:
                pipe.incrby(tok_key, tokens)
                pipe.expire(tok_key, self._window_seconds, nx=True)
                pipe.incrby(cost_key, cost_micros)
                pipe.expire(cost_key, self._window_seconds, nx=True)
                pipe.sadd(self._scopes_key, clean_scope)
                results = await await_redis(pipe.execute())
            used = int(results[0])
            total_micros = int(results[2])
            return self._status(used, total_micros)
        except (RedisError, OSError, TimeoutError, ValueError) as exc:
            logger.warning(
                "RedisTokenBudget.record failed (fail-open) for %s: %s",
                clean_scope,
                exc,
            )
            return self._fail_open_status()

    async def snapshot(self) -> dict[str, Any]:
        try:
            members = await await_redis(self._redis.smembers(self._scopes_key))
            sorted_scopes = sorted(str(m) for m in members)
            expired: list[str] = []
            active_scopes = 0
            total_tokens = 0
            total_micros = 0
            if sorted_scopes:
                async with self._redis.pipeline(transaction=False) as pipe:
                    for raw_scope in sorted_scopes:
                        pipe.get(self._tokens_key(raw_scope))
                        pipe.get(self._cost_key(raw_scope))
                    raw_values = await await_redis(pipe.execute())
                for idx, raw_scope in enumerate(sorted_scopes):
                    raw_tok = raw_values[idx * 2]
                    raw_cost = raw_values[idx * 2 + 1]
                    if raw_tok is None:
                        expired.append(raw_scope)
                        continue
                    active_scopes += 1
                    total_tokens += int(raw_tok)
                    total_micros += int(raw_cost) if raw_cost is not None else 0
            if expired:
                await await_redis(self._redis.srem(self._scopes_key, *expired))
            return {
                "active_scopes": active_scopes,
                "total_tokens": total_tokens,
                "total_estimated_cost_usd": round(total_micros / 1_000_000.0, 6),
                "limit_tokens_per_window": self._max_tokens,
            }
        except (RedisError, OSError, TimeoutError, ValueError) as exc:
            logger.warning("RedisTokenBudget.snapshot failed (fail-open): %s", exc)
            return {
                "active_scopes": 0,
                "total_tokens": 0,
                "total_estimated_cost_usd": 0.0,
                "limit_tokens_per_window": self._max_tokens,
            }


def create_token_budget(
    redis: Redis | None,
    *,
    max_tokens: int = _DEFAULT_MAX_TOKENS_PER_WINDOW,
    window_seconds: int = _DEFAULT_WINDOW_SECONDS,
) -> TokenBudget:
    """Factory creating `RedisTokenBudget` when Redis is configured, else `InMemoryTokenBudget`."""
    if redis is not None:
        return RedisTokenBudget(
            redis,
            max_tokens_per_window=max_tokens,
            window_seconds=window_seconds,
        )
    return InMemoryTokenBudget(
        max_tokens_per_window=max_tokens,
        window_seconds=window_seconds,
    )
