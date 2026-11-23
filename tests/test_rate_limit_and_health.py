"""Tests for Flux SpaceContext alignment, per-space rate limiting, and deep /health probes."""

import asyncio
from typing import Any, cast

from fastapi.testclient import TestClient
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from pydantic import SecretStr, ValidationError
from redis.exceptions import RedisError

from ai.bootstrap import _run_startup_warmup
from ai.main import create_app
from ai.orchestrator.contracts import AgentOutput, AgentRegistry, AgentSpec
from ai.orchestrator.state import RootState
from ai.runtime_builder import InfraBundle
from ai.shared.config.database import DatabaseConfig, PostgresConfig, QdrantConfig, RedisConfig
from ai.shared.config.llm import LLMConfig
from ai.shared.config.rate_limit import RateLimitConfig
from ai.shared.config.server import AppEnvironment, ServerConfig
from ai.shared.config.settings import Settings
from ai.shared.contracts.context import SpaceContext
from ai.shared.contracts.health import ComponentStatus
from ai.shared.llm.factory import ChatRunnable
from ai.shared.persistence.health import (
    PostgresCheckpointerProbe,
    QdrantProbe,
    RedisProbe,
    evaluate_health,
)
from ai.shared.persistence.rate_limiter import RedisSlidingWindowRateLimiter


def _stub_llm() -> ChatRunnable:
    def _respond(_messages: LanguageModelInput) -> AIMessage:
        return AIMessage(content="ok")

    return RunnableLambda(_respond)


def _test_settings(*, rate_limit: RateLimitConfig | None = None) -> Settings:
    return Settings(
        server=ServerConfig(
            app_env=AppEnvironment.TEST,
            internal_api_key=SecretStr("test-key"),
        ),
        llm=LLMConfig(api_key=SecretStr("test-llm-key")),
        database=DatabaseConfig(
            postgres=PostgresConfig(database_url=SecretStr("")),
            redis=RedisConfig(redis_url=SecretStr("")),
            qdrant=QdrantConfig(qdrant_url=""),
        ),
        rate_limit=rate_limit or RateLimitConfig(enabled=True, requests_per_window=30),
    )


class _StubProbe:
    def __init__(self, result: ComponentStatus, *, delay: float = 0.0) -> None:
        self._result = result
        self._delay = delay

    @property
    def name(self) -> str:
        return self._result.name

    @property
    def required(self) -> bool:
        return self._result.required

    async def check(self) -> ComponentStatus:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        return self._result


class _FakeRedis:
    def __init__(
        self,
        *,
        eval_responses: list[Any] | None = None,
        eval_error: Exception | None = None,
        ping_result: bool = True,
        ping_error: Exception | None = None,
    ) -> None:
        self._eval_responses = list(eval_responses or [])
        self._eval_error = eval_error
        self._ping_result = ping_result
        self._ping_error = ping_error
        self.eval_calls: list[tuple[Any, ...]] = []
        self.ping_calls: int = 0

    async def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> Any:
        del script
        self.eval_calls.append((numkeys, *keys_and_args))
        if self._eval_error is not None:
            raise self._eval_error
        return self._eval_responses.pop(0)

    async def ping(self) -> bool:
        self.ping_calls += 1
        if self._ping_error is not None:
            raise self._ping_error
        return self._ping_result


class _FakeQdrant:
    def __init__(self, *, error: Exception | None = None) -> None:
        self._error = error

    async def get_collections(self) -> dict[str, Any]:
        if self._error is not None:
            raise self._error
        return {"collections": []}


class _FakeConn:
    def __init__(self, *, closed: bool = False) -> None:
        self.closed = closed


class _FakePostgresSaver:
    def __init__(self, conn: Any) -> None:
        self.conn = conn


def test_space_context_masks_secret_and_ignores_legacy_wire_fields() -> None:
    ctx = SpaceContext(
        space_id="space-1",
        chat_id="chat-1",
        page_id="page-9",
        document_ids=("doc-1", "doc-2"),
        delegation_token=SecretStr("super-secret-jwt"),
        trace_id="trace-xyz",
    )
    rendered = f"{ctx!r} {ctx!s}"
    assert "super-secret-jwt" not in rendered

    configurable = ctx.to_configurable()
    assert configurable == {
        "thread_id": "chat-1",
        "space_id": "space-1",
        "chat_id": "chat-1",
        "page_id": "page-9",
        "document_ids": ["doc-1", "doc-2"],
        "trace_id": "trace-xyz",
    }
    assert "delegation_token" not in configurable
    try:
        ctx.space_id = "other"  # type: ignore[misc]
        raise AssertionError("SpaceContext must be immutable")
    except ValidationError:
        pass

    observed_configurable: dict[str, Any] = {}
    observed_context: dict[str, Any] = {}

    async def inspect_handler(state: RootState, config: RunnableConfig) -> AgentOutput:
        observed_configurable.update(config.get("configurable", {}))
        observed_context.update(state.get("context", {}))
        return AgentOutput(agent="inspector", content="inspected-ok")

    registry = AgentRegistry(
        [AgentSpec(name="inspector", description="Inspects context", handler=inspect_handler)]
    )
    router_stub = RunnableLambda(
        lambda _x: AIMessage(
            content='{"mode": "delegate", "targets": [{"agent": "inspector", "instruction": "go"}]}'
        )
    )
    app = create_app(
        settings=_test_settings(),
        llm=_stub_llm(),
        router_llm=router_stub,
        registry=registry,
    )
    headers = {"X-Internal-Key": "test-key", "Authorization": "Bearer jwt-delegation-secret"}
    with TestClient(app) as client:
        res = client.post(
            "/chat/sync",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "Inspect this page"}],
                "space_id": "space-42",
                "chat_id": "chat-99",
                "page_id": "page-7",
                "document_ids": ["doc-a", "doc-b"],
                "user_id": "legacy-user-ignored",
                "project_id": "legacy-project-ignored",
                "unknown_extra": 123,
            },
        )
    assert res.status_code == 200
    assert observed_configurable["space_id"] == "space-42"
    assert observed_configurable["chat_id"] == "chat-99"
    assert observed_configurable["page_id"] == "page-7"
    assert observed_configurable["document_ids"] == ["doc-a", "doc-b"]
    assert "user_id" not in observed_configurable
    assert "project_id" not in observed_configurable
    assert observed_context["page_id"] == "page-7"
    assert observed_context["document_ids"] == ["doc-a", "doc-b"]


def test_per_space_rate_limit_enforces_429_and_isolates_spaces() -> None:
    app = create_app(
        settings=_test_settings(
            rate_limit=RateLimitConfig(
                enabled=True,
                requests_per_window=2,
                window_seconds=60,
            )
        ),
        llm=_stub_llm(),
    )
    headers = {"X-Internal-Key": "test-key"}
    with TestClient(app) as client:
        r1 = client.post(
            "/chat/sync",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "hi 1"}],
                "space_id": "space-alpha",
                "chat_id": "c1",
            },
        )
        assert r1.status_code == 200
        assert r1.headers["X-RateLimit-Limit"] == "2"
        assert r1.headers["X-RateLimit-Remaining"] == "1"

        r2 = client.post(
            "/chat/sync",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "hi 2"}],
                "space_id": "space-alpha",
                "chat_id": "c1",
            },
        )
        assert r2.status_code == 200
        assert r2.headers["X-RateLimit-Remaining"] == "0"

        r3 = client.post(
            "/chat/sync",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "hi 3"}],
                "space_id": "space-alpha",
                "chat_id": "c1",
            },
        )
        assert r3.status_code == 429
        assert int(r3.headers["Retry-After"]) >= 1
        assert r3.json()["detail"]["space_id"] == "space-alpha"

        other_space = client.post(
            "/chat/sync",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "hi beta"}],
                "space_id": "space-beta",
                "chat_id": "c2",
            },
        )
        assert other_space.status_code == 200
        assert other_space.headers["X-RateLimit-Remaining"] == "1"


def test_disabled_rate_limiter_never_throttles() -> None:
    app = create_app(
        settings=_test_settings(
            rate_limit=RateLimitConfig(
                enabled=False,
                requests_per_window=1,
                window_seconds=60,
            )
        ),
        llm=_stub_llm(),
    )
    headers = {"X-Internal-Key": "test-key"}
    with TestClient(app) as client:
        for idx in range(3):
            res = client.post(
                "/chat/sync",
                headers=headers,
                json={
                    "messages": [{"role": "user", "content": f"msg {idx}"}],
                    "space_id": "space-unlimited",
                    "chat_id": "c1",
                },
            )
            assert res.status_code == 200


async def test_redis_sliding_window_rate_limiter_eval_and_fail_open_closed() -> None:
    fake_redis = _FakeRedis(eval_responses=[[1, 4, 60], [0, 0, 17]])
    limiter = RedisSlidingWindowRateLimiter(
        cast(Any, fake_redis),
        limit=5,
        window_seconds=60,
        fail_open=True,
    )

    ok = await limiter.check("sp-1")
    assert ok.allowed is True
    assert ok.remaining == 4
    assert fake_redis.eval_calls[0][1] == "ai:rl:space:sp-1"

    blocked = await limiter.check("sp-1")
    assert blocked.allowed is False
    assert blocked.remaining == 0
    assert blocked.reset_after_seconds == 17
    assert blocked.reason == "rate_limit_exceeded"

    broken_redis = _FakeRedis(eval_error=RedisError("redis unavailable"))
    open_limiter = RedisSlidingWindowRateLimiter(
        cast(Any, broken_redis),
        limit=5,
        window_seconds=60,
        fail_open=True,
    )
    res_open = await open_limiter.check("sp-2")
    assert res_open.allowed is True
    assert res_open.reason == "redis_unavailable_fail_open"

    closed_limiter = RedisSlidingWindowRateLimiter(
        cast(Any, broken_redis),
        limit=5,
        window_seconds=60,
        fail_open=False,
    )
    res_closed = await closed_limiter.check("sp-2")
    assert res_closed.allowed is False
    assert res_closed.reason == "redis_unavailable_fail_closed"


def test_deep_health_returns_200_or_503_based_on_required_probes() -> None:
    healthy_probes = (
        _StubProbe(ComponentStatus(name="postgres", status="ok", required=True)),
        _StubProbe(ComponentStatus(name="qdrant", status="degraded", required=False)),
    )
    app_ok = create_app(
        settings=_test_settings(),
        llm=_stub_llm(),
        health_probes=healthy_probes,
    )
    with TestClient(app_ok) as client:
        res_ok = client.get("/health")
    assert res_ok.status_code == 200
    assert res_ok.json()["status"] == "ok"

    degraded_probes = (
        _StubProbe(ComponentStatus(name="postgres", status="ok", required=True)),
        _StubProbe(
            ComponentStatus(
                name="redis",
                status="degraded",
                required=True,
                detail="connection refused",
            )
        ),
    )
    app_bad = create_app(
        settings=_test_settings(),
        llm=_stub_llm(),
        health_probes=degraded_probes,
    )
    with TestClient(app_bad) as client:
        res_bad = client.get("/health")
    assert res_bad.status_code == 503
    body_bad = res_bad.json()
    assert body_bad["status"] == "degraded"
    assert body_bad["components"][1]["detail"] == "connection refused"


async def test_concrete_infrastructure_probes_and_timeout_handling() -> None:
    mem_probe = PostgresCheckpointerProbe(MemorySaver(), configured=False)
    assert (await mem_probe.check()).status == "disabled"

    pg_ok = PostgresCheckpointerProbe(
        cast(Any, _FakePostgresSaver(_FakeConn(closed=False))),
        configured=True,
    )
    assert (await pg_ok.check()).status == "ok"

    pg_closed = PostgresCheckpointerProbe(
        cast(Any, _FakePostgresSaver(_FakeConn(closed=True))),
        configured=True,
    )
    assert (await pg_closed.check()).status == "degraded"

    redis_ok = RedisProbe(cast(Any, _FakeRedis(ping_result=True)), configured=True)
    assert (await redis_ok.check()).status == "ok"

    redis_err = RedisProbe(
        cast(Any, _FakeRedis(ping_error=RedisError("timeout"))),
        configured=True,
    )
    qdrant_ok = QdrantProbe(cast(Any, _FakeQdrant()), configured=True)
    assert (await qdrant_ok.check()).status == "ok"

    slow_probe = _StubProbe(
        ComponentStatus(name="qdrant", status="ok", required=True),
        delay=0.3,
    )
    overall, statuses = await evaluate_health(
        (pg_ok, redis_err, slow_probe),
        timeout_seconds=0.05,
    )
    assert overall == "degraded"
    assert statuses[0].status == "ok"
    assert statuses[1].status == "degraded"
    assert "timeout" in statuses[1].detail.lower()
    assert statuses[2].status == "degraded"
    assert "timed out" in statuses[2].detail.lower()


async def test_startup_warmup_invokes_redis_and_knowledge_and_fails_open() -> None:
    class _WarmupKnowledge:
        def __init__(self, *, delay: float = 0.0, error: Exception | None = None) -> None:
            self._delay = delay
            self._error = error
            self.ready_calls = 0

        async def ensure_ready(self) -> None:
            self.ready_calls += 1
            if self._delay > 0:
                await asyncio.sleep(self._delay)
            if self._error is not None:
                raise self._error

    def _infra(redis: Any, knowledge: Any) -> InfraBundle:
        return InfraBundle(
            http=cast(Any, None),
            checkpointer=MemorySaver(),
            redis=redis,
            qdrant=None,
            memory_store=cast(Any, None),
            knowledge_store=cast(Any, knowledge),
        )

    redis_ok = _FakeRedis(ping_result=True)
    kb_ok = _WarmupKnowledge()
    await _run_startup_warmup(_infra(redis_ok, kb_ok), timeout_seconds=0.5)
    assert redis_ok.ping_calls == 1
    assert kb_ok.ready_calls == 1

    # Fail-open when backends raise errors or time out
    redis_fail = _FakeRedis(ping_error=RedisError("connection lost"))
    kb_slow = _WarmupKnowledge(delay=0.3)
    await _run_startup_warmup(_infra(redis_fail, kb_slow), timeout_seconds=0.05)
    assert redis_fail.ping_calls == 1
    assert kb_slow.ready_calls == 1
