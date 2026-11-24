"""Unit and FastAPI integration tests for Observability, InMemoryTokenBudget & /health/resilience."""

from __future__ import annotations

from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr
from redis.asyncio import Redis
from redis.exceptions import RedisError

from ai.main import create_app
from ai.shared.config.database import DatabaseConfig, PostgresConfig, QdrantConfig, RedisConfig
from ai.shared.config.embedding import EmbeddingConfig
from ai.shared.config.llm import LLMConfig
from ai.shared.config.server import AppEnvironment, ServerConfig
from ai.shared.config.settings import Settings
from ai.shared.contracts.run_control import RunUsage
from ai.shared.llm.factory import ChatRunnable, ContextAwareChatRunnable
from ai.shared.resilience.observability import (
    InMemoryTokenBudget,
    LLMCallMetrics,
    RedisTokenBudget,
    TokenBudget,
    ToolExecutionMetrics,
    create_token_budget,
)
from ai.shared.resilience.prometheus import render_prometheus_metrics


def _test_settings() -> Settings:
    return Settings(
        server=ServerConfig(
            app_env=AppEnvironment.TEST,
            internal_api_key=SecretStr("obs-test-key"),
        ),
        llm=LLMConfig(api_key=SecretStr("test-llm-key")),
        embedding=EmbeddingConfig(dimensions=64, collection_name="test_knowledge"),
        database=DatabaseConfig(
            postgres=PostgresConfig(database_url=SecretStr("")),
            redis=RedisConfig(redis_url=SecretStr("")),
            qdrant=QdrantConfig(qdrant_url=""),
        ),
    )


def test_tool_execution_metrics_counters_and_avg_latency() -> None:
    metrics = ToolExecutionMetrics()
    assert metrics.snapshot() == {}

    metrics.record_call("insert_experiment_table", latency_ms=120.0, ok=True, cache_hit=False)
    metrics.record_call("insert_experiment_table", latency_ms=6.0, ok=True, cache_hit=True)
    metrics.record_call("insert_experiment_table", latency_ms=54.0, ok=False, cache_hit=False)
    metrics.record_call("search_items", latency_ms=30.0, ok=True, cache_hit=False)

    snap = metrics.snapshot()
    assert snap["insert_experiment_table"] == {
        "calls": 3,
        "errors": 1,
        "cache_hits": 1,
        "total_latency_ms": 180.0,
        "avg_latency_ms": 60.0,
    }
    assert snap["search_items"] == {
        "calls": 1,
        "errors": 0,
        "cache_hits": 0,
        "total_latency_ms": 30.0,
        "avg_latency_ms": 30.0,
    }


async def test_in_memory_token_budget_cost_quota_and_sliding_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"t": 1000.0}
    monkeypatch.setattr("ai.shared.resilience.observability.time.monotonic", lambda: now["t"])

    guard = InMemoryTokenBudget(
        max_tokens_per_window=1000,
        window_seconds=60,
        cost_per_1k_input_usd=0.001,
        cost_per_1k_output_usd=0.002,
    )

    initial = await guard.check("space-a")
    assert initial.allowed is True
    assert initial.used_tokens == 0
    assert initial.remaining_tokens == 1000
    assert initial.estimated_cost_usd == 0.0

    # Record 600 tokens (400 in = $0.0004, 200 out = $0.0004 => $0.0008)
    s1 = await guard.record(
        "space-a",
        RunUsage(input_tokens=400, output_tokens=200, total_tokens=600),
    )
    assert s1.allowed is True
    assert s1.used_tokens == 600
    assert s1.remaining_tokens == 400
    assert s1.estimated_cost_usd == pytest.approx(0.0008)

    # Record another 400 tokens -> hits exact 1000 limit => allowed=False
    s2 = await guard.record(
        "space-a",
        RunUsage(input_tokens=200, output_tokens=200, total_tokens=400),
    )
    assert s2.allowed is False
    assert s2.used_tokens == 1000
    assert s2.remaining_tokens == 0
    assert s2.estimated_cost_usd == pytest.approx(0.0014)
    assert (await guard.check("space-a")).allowed is False

    # Other scope remains isolated and allowed
    assert (await guard.check("space-b")).allowed is True
    await guard.record("space-b", RunUsage(input_tokens=100, output_tokens=50, total_tokens=150))

    agg = await guard.snapshot()
    assert agg["active_scopes"] == 2
    assert agg["total_tokens"] == 1150
    assert agg["limit_tokens_per_window"] == 1000

    # Advance clock past 60s window -> expired events drop cleanly
    now["t"] = 1061.0
    after_expiry = await guard.check("space-a")
    assert after_expiry.allowed is True
    assert after_expiry.used_tokens == 0
    assert after_expiry.remaining_tokens == 1000
    assert (await guard.snapshot())["active_scopes"] == 0


async def test_create_token_budget_factory_and_in_memory_adapter() -> None:
    assert isinstance(create_token_budget(None), InMemoryTokenBudget)
    redis_client = Redis.from_url("redis://localhost:6379", decode_responses=True)
    assert isinstance(create_token_budget(redis_client), RedisTokenBudget)

    budget: TokenBudget = create_token_budget(None, max_tokens=500, window_seconds=60)
    assert isinstance(budget, TokenBudget)

    s0 = await budget.check("space-mem")
    assert s0.allowed is True
    assert s0.remaining_tokens == 500

    s1 = await budget.record(
        "space-mem", RunUsage(input_tokens=300, output_tokens=200, total_tokens=500)
    )
    assert s1.allowed is False
    assert s1.used_tokens == 500
    assert s1.remaining_tokens == 0

    snap = await budget.snapshot()
    assert snap["active_scopes"] == 1
    assert snap["total_tokens"] == 500


async def test_redis_token_budget_adapter_and_fail_open() -> None:
    class _FakeAsyncRedis:
        def __init__(self) -> None:
            self.kv: dict[str, int] = {}
            self.sets: dict[str, set[str]] = {}
            self.ttls: dict[str, int] = {}
            self.fail = False

        def _guard(self) -> None:
            if self.fail:
                raise RedisError("simulated redis outage")

        async def get(self, key: str) -> str | None:
            self._guard()
            val = self.kv.get(key)
            return str(val) if val is not None else None

        async def incrby(self, key: str, amount: int) -> int:
            self._guard()
            self.kv[key] = self.kv.get(key, 0) + amount
            return self.kv[key]

        async def expire(self, key: str, seconds: int, *, nx: bool = False) -> bool:
            self._guard()
            if nx and key in self.ttls:
                return False
            self.ttls[key] = seconds
            return True

        async def sadd(self, key: str, *values: str) -> int:
            self._guard()
            bucket = self.sets.setdefault(key, set())
            before = len(bucket)
            bucket.update(values)
            return len(bucket) - before

        async def smembers(self, key: str) -> set[str]:
            self._guard()
            return set(self.sets.get(key, set()))

        async def srem(self, key: str, *values: str) -> int:
            self._guard()
            bucket = self.sets.get(key, set())
            removed = sum(1 for v in values if v in bucket)
            for v in values:
                bucket.discard(v)
            return removed

    fake = _FakeAsyncRedis()
    budget = RedisTokenBudget(
        cast(Redis, fake),
        max_tokens_per_window=1000,
        window_seconds=3600,
        cost_per_1k_input_usd=0.001,
        cost_per_1k_output_usd=0.002,
    )

    # First record sets tokens + cost + TTL
    r1 = await budget.record(
        "space-r1", RunUsage(input_tokens=400, output_tokens=200, total_tokens=600)
    )
    assert r1.allowed is True
    assert r1.used_tokens == 600
    assert r1.estimated_cost_usd == pytest.approx(0.0008)
    assert fake.ttls["flux:budget:tokens:space-r1"] == 3600

    # Second record pushes to 1000 -> allowed=False
    r2 = await budget.record(
        "space-r1", RunUsage(input_tokens=200, output_tokens=200, total_tokens=400)
    )
    assert r2.allowed is False
    assert r2.used_tokens == 1000
    assert r2.remaining_tokens == 0
    assert (await budget.check("space-r1")).allowed is False

    # Snapshot reflects active scope totals
    snap = await budget.snapshot()
    assert snap == {
        "active_scopes": 1,
        "total_tokens": 1000,
        "total_estimated_cost_usd": pytest.approx(0.0014),
        "limit_tokens_per_window": 1000,
    }

    # Simulated Redis outage -> fail-open (allowed=True, does not crash)
    fake.fail = True
    assert (await budget.check("space-r1")).allowed is True
    assert (
        await budget.record(
            "space-r1", RunUsage(input_tokens=10, output_tokens=10, total_tokens=20)
        )
    ).allowed is True
    assert (await budget.snapshot())["active_scopes"] == 0


async def test_resilience_health_endpoint_auth_and_payload() -> None:
    def _dummy_llm(_prompt: LanguageModelInput) -> AIMessage:
        return AIMessage(content="ok")

    app = create_app(
        settings=_test_settings(),
        llm=RunnableLambda(_dummy_llm),
    )
    with TestClient(app) as client:
        # 1. Missing or invalid X-Internal-Key -> 401
        unauth = client.get("/health/resilience")
        assert unauth.status_code == 401

        bad_auth = client.get("/health/resilience", headers={"X-Internal-Key": "wrong-key"})
        assert bad_auth.status_code == 401

        # 2. Mutate wired state to confirm live state reflection
        app.state.tool_metrics.record_call("github_fetch", latency_ms=42.5, ok=True)
        app.state.llm_metrics.record_call(
            "agent",
            "deepseek-v3.2",
            latency_ms=250.0,
            ok=True,
            input_tokens=100,
            output_tokens=50,
        )
        await app.state.token_budget.record(
            "space-live", RunUsage(input_tokens=500, output_tokens=100, total_tokens=600)
        )

        # 3. Valid X-Internal-Key -> 200 with expected schema
        res = client.get("/health/resilience", headers={"X-Internal-Key": "obs-test-key"})
        assert res.status_code == 200
        body = res.json()
        assert set(body.keys()) == {
            "circuit_breakers",
            "tool_metrics",
            "llm_metrics",
            "ingestion_queue",
            "token_budget",
        }
        assert body["circuit_breakers"].get("github") == "closed"
        assert body["tool_metrics"]["github_fetch"]["calls"] == 1
        assert body["llm_metrics"]["agent:deepseek-v3.2"]["calls"] == 1
        assert body["ingestion_queue"]["max_queue_size"] >= 1
        assert body["token_budget"]["total_tokens"] == 600

        # 4. Prometheus /metrics requires auth and emits 0.0.4 text format
        metrics_unauth = client.get("/metrics")
        assert metrics_unauth.status_code == 401

        metrics_res = client.get("/metrics", headers={"X-Internal-Key": "obs-test-key"})
        assert metrics_res.status_code == 200
        assert metrics_res.headers["content-type"].startswith("text/plain; version=0.0.4")
        metrics_text = metrics_res.text
        assert 'ai_tool_calls_total{tool="github_fetch"} 1' in metrics_text
        assert 'ai_llm_calls_total{role="agent",model="deepseek-v3.2"} 1' in metrics_text
        assert (
            'ai_llm_tokens_total{role="agent",model="deepseek-v3.2",direction="in"} 100'
            in metrics_text
        )
        assert 'ai_circuit_breaker_state{service="github"} 0' in metrics_text
        assert "ai_ingestion_queue_active 0" in metrics_text
        assert "ai_ingestion_queue_capacity " in metrics_text

        # 5. Existing /health contract remains unchanged
        base_health = client.get("/health")
        assert base_health.status_code == 200
        assert set(base_health.json().keys()) == {"status", "app_env", "components"}


async def test_chat_enforces_and_records_token_budget() -> None:
    def _dummy_llm(_prompt: LanguageModelInput) -> AIMessage:
        return AIMessage(
            content="answer",
            usage_metadata={"input_tokens": 120, "output_tokens": 80, "total_tokens": 200},
        )

    app = create_app(
        settings=_test_settings(),
        llm=RunnableLambda(_dummy_llm),
    )
    headers = {"X-Internal-Key": "obs-test-key"}
    with TestClient(app) as client:
        # Each turn runs supervisor router (200 tok) + direct response (200 tok) = 400 tokens.
        app.state.token_budget = InMemoryTokenBudget(max_tokens_per_window=500)

        # 1. First sync chat consumes 400 tokens (< 500 limit) and records usage
        ok1 = client.post(
            "/chat/sync",
            headers=headers,
            json={
                "space_id": "space-budget-1",
                "stateless": True,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert ok1.status_code == 200
        quota_after_1 = await app.state.token_budget.check("space-budget-1")
        assert quota_after_1.used_tokens == 400
        assert quota_after_1.allowed is True

        # 2. Second sync chat pushes used_tokens to 800 (>= 500) -> allowed becomes False
        ok2 = client.post(
            "/chat/sync",
            headers=headers,
            json={
                "space_id": "space-budget-1",
                "stateless": True,
                "messages": [{"role": "user", "content": "Second question"}],
            },
        )
        assert ok2.status_code == 200
        quota_after_2 = await app.state.token_budget.check("space-budget-1")
        assert quota_after_2.allowed is False

        # 3. Third chat request is blocked at the gate with 429 token_budget_exceeded
        blocked = client.post(
            "/chat/sync",
            headers=headers,
            json={
                "space_id": "space-budget-1",
                "stateless": True,
                "messages": [{"role": "user", "content": "Blocked question"}],
            },
        )
        assert blocked.status_code == 429
        assert blocked.json()["detail"]["error"] == "token_budget_exceeded"

        # 4. BYOK request (with context.llm_override) bypasses platform token budget!
        byok_ok = client.post(
            "/chat/sync",
            headers=headers,
            json={
                "space_id": "space-budget-1",
                "stateless": True,
                "messages": [{"role": "user", "content": "BYOK question"}],
                "context": {
                    "llm_override": {
                        "api_key": "sk-user-byok-key",
                        "model": "gpt-4o",
                        "base_url": "https://api.openai.com/v1",
                    }
                },
            },
        )
        assert byok_ok.status_code == 200


def test_llm_call_metrics_counters_and_avg_latency() -> None:
    metrics = LLMCallMetrics()
    assert metrics.snapshot() == {}

    metrics.record_call(
        "agent",
        "deepseek-v3.2",
        latency_ms=120.0,
        ok=True,
        input_tokens=300,
        output_tokens=150,
    )
    metrics.record_call(
        "agent",
        "deepseek-v3.2",
        latency_ms=80.0,
        ok=False,
        input_tokens=100,
        output_tokens=0,
    )
    metrics.record_call(
        "",
        "",
        latency_ms=-5.0,
        ok=True,
        input_tokens=-10,
        output_tokens=20,
    )

    snap = metrics.snapshot()
    assert snap["agent:deepseek-v3.2"] == {
        "role": "agent",
        "model": "deepseek-v3.2",
        "calls": 2,
        "errors": 1,
        "input_tokens": 400,
        "output_tokens": 150,
        "total_latency_ms": 200.0,
        "avg_latency_ms": 100.0,
    }
    assert snap["default:unknown"] == {
        "role": "default",
        "model": "unknown",
        "calls": 1,
        "errors": 0,
        "input_tokens": 0,
        "output_tokens": 20,
        "total_latency_ms": 0.0,
        "avg_latency_ms": 0.0,
    }


def test_render_prometheus_metrics_escapes_labels_and_formats_states() -> None:
    text = render_prometheus_metrics(
        tool_metrics={
            'weird"tool\\n\n': {
                "calls": 2,
                "errors": 1,
                "cache_hits": 1,
                "total_latency_ms": 15.5,
            }
        },
        llm_metrics={
            "router:deepseek-v4-flash": {
                "role": "router",
                "model": "deepseek-v4-flash",
                "calls": 3,
                "errors": 0,
                "input_tokens": 250,
                "output_tokens": 90,
                "total_latency_ms": 420.0,
            }
        },
        circuit_breakers={
            "qdrant": "closed",
            "github": "half_open",
            "web": "open",
        },
        ingestion_queue={"active_jobs": 2, "max_queue_size": 50},
    )

    assert 'ai_tool_calls_total{tool="weird\\"tool\\\\n\\n"} 2' in text
    assert 'ai_llm_calls_total{role="router",model="deepseek-v4-flash"} 3' in text
    assert 'ai_llm_tokens_total{role="router",model="deepseek-v4-flash",direction="in"} 250' in text
    assert 'ai_llm_tokens_total{role="router",model="deepseek-v4-flash",direction="out"} 90' in text
    assert 'ai_circuit_breaker_state{service="qdrant"} 0' in text
    assert 'ai_circuit_breaker_state{service="github"} 1' in text
    assert 'ai_circuit_breaker_state{service="web"} 2' in text
    assert "ai_ingestion_queue_active 2" in text
    assert "ai_ingestion_queue_capacity 50" in text
    assert text.endswith("\n")


async def test_context_aware_runnable_records_ainvoke_astream_and_errors() -> None:
    class _StubTarget:
        def __init__(self, *, fail: bool = False) -> None:
            self._fail = fail

        async def ainvoke(self, _input: LanguageModelInput, **_kwargs: Any) -> AIMessage:
            if self._fail:
                raise RuntimeError("upstream 503")
            return AIMessage(
                content="ok",
                usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
                response_metadata={"model_name": "gemini-2.5-flash"},
            )

        async def astream(self, _input: LanguageModelInput, **_kwargs: Any) -> Any:
            yield AIMessageChunk(content="hi ")
            yield AIMessageChunk(
                content="there",
                usage_metadata={"input_tokens": 15, "output_tokens": 8, "total_tokens": 23},
                response_metadata={"model_name": "deepseek-v3.2"},
            )

    cfg = LLMConfig(
        api_key=SecretStr("sk-test"),
        model="gemini-2.5-flash",
        router_model="deepseek-v4-flash",
        agent_model="deepseek-v3.2",
        router_api_key=SecretStr(""),
        agent_api_key=SecretStr(""),
    )
    metrics = LLMCallMetrics()

    default_proxy = ContextAwareChatRunnable(
        cast(ChatRunnable, _StubTarget()),
        config=cfg,
        role="default",
        metrics=metrics,
    )
    res = await default_proxy.ainvoke("ping")
    assert res.content == "ok"

    agent_proxy = ContextAwareChatRunnable(
        cast(ChatRunnable, _StubTarget()),
        config=cfg,
        role="agent",
        metrics=metrics,
    )
    chunks = [chunk.content async for chunk in agent_proxy.astream("ping")]
    assert chunks == ["hi ", "there"]

    router_proxy = ContextAwareChatRunnable(
        cast(ChatRunnable, _StubTarget(fail=True)),
        config=cfg,
        role="router",
        metrics=metrics,
    )
    with pytest.raises(RuntimeError, match="upstream 503"):
        await router_proxy.ainvoke("route")

    snap = metrics.snapshot()
    assert snap["default:gemini-2.5-flash"]["calls"] == 1
    assert snap["default:gemini-2.5-flash"]["errors"] == 0
    assert snap["default:gemini-2.5-flash"]["input_tokens"] == 20
    assert snap["default:gemini-2.5-flash"]["output_tokens"] == 10

    assert snap["agent:deepseek-v3.2"]["calls"] == 1
    assert snap["agent:deepseek-v3.2"]["errors"] == 0
    assert snap["agent:deepseek-v3.2"]["input_tokens"] == 15
    assert snap["agent:deepseek-v3.2"]["output_tokens"] == 8

    assert snap["router:deepseek-v4-flash"]["calls"] == 1
    assert snap["router:deepseek-v4-flash"]["errors"] == 1
    assert snap["router:deepseek-v4-flash"]["input_tokens"] == 0
