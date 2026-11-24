"""Seam tests for pipelined RedisTokenBudget round-trip batching."""

from __future__ import annotations

from typing import Any, Self, cast

import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError

from ai.shared.contracts.run_control import RunUsage
from ai.shared.resilience.observability import RedisTokenBudget


class _PipelineFakeRedis:
    """Async Redis fake that supports `pipeline(transaction=False)` and counts RTTs."""

    def __init__(self) -> None:
        self.kv: dict[str, int] = {}
        self.sets: dict[str, set[str]] = {}
        self.ttls: dict[str, int] = {}
        self.fail = False
        self.round_trips = 0

    def _guard(self) -> None:
        if self.fail:
            raise RedisError("simulated pipeline outage")

    def pipeline(self, *, transaction: bool = True) -> _FakePipeline:
        return _FakePipeline(self, transaction=transaction)

    async def get(self, key: str) -> str | None:
        self.round_trips += 1
        self._guard()
        val = self.kv.get(key)
        return str(val) if val is not None else None

    async def incrby(self, key: str, amount: int) -> int:
        self.round_trips += 1
        self._guard()
        self.kv[key] = self.kv.get(key, 0) + amount
        return self.kv[key]

    async def expire(self, key: str, seconds: int, *, nx: bool = False) -> bool:
        self.round_trips += 1
        self._guard()
        if nx and key in self.ttls:
            return False
        self.ttls[key] = seconds
        return True

    async def sadd(self, key: str, *values: str) -> int:
        self.round_trips += 1
        self._guard()
        bucket = self.sets.setdefault(key, set())
        before = len(bucket)
        bucket.update(values)
        return len(bucket) - before

    async def smembers(self, key: str) -> set[str]:
        self.round_trips += 1
        self._guard()
        return set(self.sets.get(key, set()))

    async def srem(self, key: str, *values: str) -> int:
        self.round_trips += 1
        self._guard()
        bucket = self.sets.get(key, set())
        removed = sum(1 for v in values if v in bucket)
        for v in values:
            bucket.discard(v)
        return removed


class _FakePipeline:
    def __init__(self, redis: _PipelineFakeRedis, *, transaction: bool) -> None:
        self._redis = redis
        self.transaction = transaction
        self._ops: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    def get(self, key: str) -> _FakePipeline:
        self._ops.append(("get", (key,), {}))
        return self

    def incrby(self, key: str, amount: int) -> _FakePipeline:
        self._ops.append(("incrby", (key, amount), {}))
        return self

    def expire(self, key: str, seconds: int, *, nx: bool = False) -> _FakePipeline:
        self._ops.append(("expire", (key, seconds), {"nx": nx}))
        return self

    def sadd(self, key: str, *values: str) -> _FakePipeline:
        self._ops.append(("sadd", (key, *values), {}))
        return self

    async def execute(self) -> list[Any]:
        self._redis.round_trips += 1
        self._redis._guard()
        results: list[Any] = []
        for op, args, kwargs in self._ops:
            if op == "get":
                val = self._redis.kv.get(args[0])
                results.append(str(val) if val is not None else None)
            elif op == "incrby":
                key, amount = args
                self._redis.kv[key] = self._redis.kv.get(key, 0) + amount
                results.append(self._redis.kv[key])
            elif op == "expire":
                key, seconds = args
                nx = bool(kwargs.get("nx", False))
                if nx and key in self._redis.ttls:
                    results.append(False)
                else:
                    self._redis.ttls[key] = seconds
                    results.append(True)
            elif op == "sadd":
                key, *values = args
                bucket = self._redis.sets.setdefault(key, set())
                before = len(bucket)
                bucket.update(values)
                results.append(len(bucket) - before)
        self._ops.clear()
        return results


async def test_redis_token_budget_batches_record_and_check_into_single_round_trip() -> None:
    fake = _PipelineFakeRedis()
    budget = RedisTokenBudget(
        cast(Redis, fake),
        max_tokens_per_window=1000,
        window_seconds=3600,
        cost_per_1k_input_usd=0.001,
        cost_per_1k_output_usd=0.002,
    )

    # record() should execute all updates in 1 pipeline round-trip (not 5 sequential RTTs)
    fake.round_trips = 0
    r1 = await budget.record(
        "space-p1", RunUsage(input_tokens=400, output_tokens=200, total_tokens=600)
    )
    assert fake.round_trips == 1
    assert r1.allowed is True
    assert r1.used_tokens == 600
    assert r1.remaining_tokens == 400
    assert r1.estimated_cost_usd == pytest.approx(0.0008)
    assert fake.ttls["flux:budget:tokens:space-p1"] == 3600
    assert fake.ttls["flux:budget:cost:space-p1"] == 3600

    # check() should read both tokens and cost in 1 pipeline round-trip (not 2 sequential GETs)
    fake.round_trips = 0
    c1 = await budget.check("space-p1")
    assert fake.round_trips == 1
    assert c1.allowed is True
    assert c1.used_tokens == 600
    assert c1.estimated_cost_usd == pytest.approx(0.0008)

    # Second scope + snapshot() should pipeline per-scope reads instead of 2*N sequential GETs
    await budget.record("space-p2", RunUsage(input_tokens=100, output_tokens=100, total_tokens=200))
    fake.round_trips = 0
    snap = await budget.snapshot()
    assert fake.round_trips == 2  # 1 smembers + 1 pipelined batch of GETs
    assert snap["active_scopes"] == 2
    assert snap["total_tokens"] == 800

    # Outage during pipelined execute fails open
    fake.fail = True
    assert (await budget.check("space-p1")).allowed is True
    assert (
        await budget.record("space-p1", RunUsage(input_tokens=10, output_tokens=5, total_tokens=15))
    ).allowed is True
    assert (await budget.snapshot())["active_scopes"] == 0
