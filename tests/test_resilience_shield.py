"""Unit tests for Global Resilience Shield (CircuitBreaker, ToolCache, execute_tool_safely)."""

from __future__ import annotations

import base64

import httpx
import pytest
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableWithFallbacks
from pydantic import SecretStr
from redis.asyncio import Redis

from ai.agents.editor.workflow.prompts import build_planner_prompt as build_editor_planner_prompt
from ai.agents.library.workflow.prompts import build_planner_prompt as build_library_planner_prompt
from ai.agents.project.workflow.prompts import build_planner_prompt as build_project_planner_prompt
from ai.orchestrator.contracts import AgentRegistry, AgentSpec
from ai.orchestrator.planner import LLMSupervisorPlanner
from ai.orchestrator.prompts import build_supervisor_prompt
from ai.orchestrator.state import RootState
from ai.shared.config.llm import LLMConfig
from ai.shared.github.client import HttpGitHubArtifactClient
from ai.shared.github.contracts import GitHubArtifactError
from ai.shared.llm.factory import (
    ContextAwareChatRunnable,
    LLMOverride,
    bind_llm_override,
    build_chat_model,
    parse_llm_override,
)
from ai.shared.resilience.cache import (
    InMemoryToolCache,
    RedisToolCache,
    build_cache_key,
    create_tool_cache,
)
from ai.shared.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry
from ai.shared.resilience.contracts import (
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
)
from ai.shared.resilience.tool_runner import execute_tool_safely


async def test_circuit_breaker_state_transitions_and_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"t": 100.0}
    monkeypatch.setattr(
        "ai.shared.resilience.circuit_breaker.time.monotonic", lambda: now["t"]
    )

    breaker = CircuitBreaker(
        "github_api",
        CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout_seconds=10.0,
            half_open_max_calls=1,
        ),
    )
    assert breaker.state == CircuitState.CLOSED

    # 2 failures keep circuit CLOSED
    await breaker.allow_request()
    await breaker.record_failure()
    await breaker.allow_request()
    await breaker.record_failure()
    assert breaker.state == CircuitState.CLOSED

    # 3rd failure trips circuit to OPEN
    await breaker.allow_request()
    await breaker.record_failure()
    assert breaker.state == CircuitState.OPEN

    # Requests short-circuit immediately while OPEN
    with pytest.raises(CircuitOpenError) as exc_info:
        await breaker.allow_request()
    assert exc_info.value.service_name == "github_api"
    assert exc_info.value.retry_after_seconds == pytest.approx(10.0)

    # Advance clock past recovery_timeout -> transitions to HALF_OPEN
    now["t"] = 111.0
    assert breaker.state == CircuitState.HALF_OPEN

    # Allow 1 probe call in HALF_OPEN, block 2nd concurrent call
    await breaker.allow_request()
    with pytest.raises(CircuitOpenError):
        await breaker.allow_request()

    # Probe failure re-opens circuit
    await breaker.record_failure()
    assert breaker.state == CircuitState.OPEN

    # Advance clock again, probe succeeds -> recovers to CLOSED
    now["t"] = 122.0
    assert breaker.state == CircuitState.HALF_OPEN
    await breaker.allow_request()
    await breaker.record_success()
    assert breaker.state == CircuitState.CLOSED

    # Subsequent calls flow freely
    await breaker.allow_request()


async def test_circuit_breaker_registry_isolation_and_snapshot() -> None:
    registry = CircuitBreakerRegistry(
        CircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=30.0)
    )
    gh = registry.get_or_create("github")
    be = registry.get_or_create("backend")
    assert be is not gh
    assert registry.get_or_create("github") is gh

    await gh.record_failure()
    await gh.record_failure()

    snap = registry.snapshot()
    assert snap == {"github": "open", "backend": "closed"}


def test_build_cache_key_is_deterministic() -> None:
    k1 = build_cache_key("github", {"repo": "a/b", "path": "x.csv", "ref": "main"})
    k2 = build_cache_key("github", {"ref": "main", "path": "x.csv", "repo": "a/b"})
    k3 = build_cache_key("github", {"ref": "dev", "path": "x.csv", "repo": "a/b"})

    assert k1 == k2
    assert k1 != k3
    assert k1.startswith("github:")


async def test_in_memory_tool_cache_ttl_delete_and_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"t": 10.0}
    monkeypatch.setattr("ai.shared.resilience.cache.time.monotonic", lambda: now["t"])

    cache = InMemoryToolCache(max_entries=2)
    await cache.set("k1", {"v": 1}, ttl_seconds=5)
    assert await cache.get("k1") == {"v": 1}

    # Eviction when exceeding max_entries
    await cache.set("k2", {"v": 2}, ttl_seconds=5)
    await cache.set("k3", {"v": 3}, ttl_seconds=5)
    assert await cache.get("k1") is None
    assert await cache.get("k2") == {"v": 2}
    assert await cache.get("k3") == {"v": 3}

    # Explicit delete
    await cache.delete("k2")
    assert await cache.get("k2") is None

    # TTL expiration
    now["t"] = 16.0
    assert await cache.get("k3") is None


async def test_execute_tool_safely_with_circuit_breaker_and_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = {"t": 50.0}
    monkeypatch.setattr(
        "ai.shared.resilience.circuit_breaker.time.monotonic", lambda: now["t"]
    )
    monkeypatch.setattr("ai.shared.resilience.cache.time.monotonic", lambda: now["t"])

    cache = InMemoryToolCache()
    breaker = CircuitBreaker(
        "downstream",
        CircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=10.0),
    )

    calls = {"n": 0, "should_fail": False}

    async def _sample_tool(x: int) -> dict[str, int]:
        calls["n"] += 1
        if calls["should_fail"]:
            raise OSError("network down")
        return {"res": x * 2}

    key = build_cache_key("sample", {"x": 5})

    # 1. Cache miss -> invokes tool and caches result
    res1 = await execute_tool_safely(
        _sample_tool,
        5,
        max_retries=0,
        circuit_breaker=breaker,
        cache=cache,
        cache_key=key,
        cache_ttl_seconds=60,
    )
    assert res1 == {"res": 10}
    assert calls["n"] == 1

    # 2. Cache hit -> skips tool invocation completely
    res2 = await execute_tool_safely(
        _sample_tool,
        5,
        max_retries=0,
        circuit_breaker=breaker,
        cache=cache,
        cache_key=key,
    )
    assert res2 == {"res": 10}
    assert calls["n"] == 1

    # 3. Failing uncached calls trip the CircuitBreaker
    calls["should_fail"] = True
    key_fail = build_cache_key("sample", {"x": 9})

    with pytest.raises(OSError):
        await execute_tool_safely(
            _sample_tool,
            9,
            max_retries=0,
            circuit_breaker=breaker,
            cache=cache,
            cache_key=key_fail,
        )
    assert breaker.state == CircuitState.CLOSED

    with pytest.raises(OSError):
        await execute_tool_safely(
            _sample_tool,
            9,
            max_retries=0,
            circuit_breaker=breaker,
            cache=cache,
            cache_key=key_fail,
        )
    assert breaker.state == CircuitState.OPEN
    assert calls["n"] == 3

    # 4. Subsequent call short-circuits without touching _sample_tool
    with pytest.raises(CircuitOpenError):
        await execute_tool_safely(
            _sample_tool,
            9,
            max_retries=0,
            circuit_breaker=breaker,
            cache=cache,
            cache_key=key_fail,
        )
    assert calls["n"] == 3

    # 5. Advance clock & fix service -> breaker recovers to CLOSED
    now["t"] = 61.0
    calls["should_fail"] = False
    res3 = await execute_tool_safely(
        _sample_tool,
        9,
        max_retries=0,
        circuit_breaker=breaker,
        cache=cache,
        cache_key=key_fail,
    )
    assert res3 == {"res": 18}
    assert breaker.state == CircuitState.CLOSED
    assert calls["n"] == 4


def test_create_tool_cache_factory() -> None:
    assert isinstance(create_tool_cache(None), InMemoryToolCache)
    redis_client = Redis.from_url("redis://localhost:6379", decode_responses=True)
    assert isinstance(create_tool_cache(redis_client), RedisToolCache)


async def test_github_artifact_client_cache_and_circuit_breaker() -> None:
    sample_csv = "model,acc\nFlux,96.4\n"
    encoded = base64.b64encode(sample_csv.encode("utf-8")).decode("utf-8")
    http_calls = {"count": 0, "mode": "ok"}

    def _handler(req: httpx.Request) -> httpx.Response:
        http_calls["count"] += 1
        if http_calls["mode"] == "404":
            return httpx.Response(404, json={"message": "Not Found"})
        if http_calls["mode"] == "500":
            return httpx.Response(500, json={"message": "Internal Server Error"})
        if req.url.path.endswith("/commits/main"):
            return httpx.Response(
                200,
                json={"sha": "b" * 40},
            )
        return httpx.Response(
            200,
            json={
                "name": "eval.csv",
                "path": "eval.csv",
                "sha": "blob-1",
                "size": len(sample_csv),
                "encoding": "base64",
                "content": encoded,
            },
        )

    cache = InMemoryToolCache()
    breaker = CircuitBreaker(
        "github",
        CircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=30.0),
    )

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as http_client:
        gh = HttpGitHubArtifactClient(
            http_client=http_client,
            circuit_breaker=breaker,
            cache=cache,
            cache_ttl_seconds=60,
        )

        # 1. First fetch hits HTTP (contents + commit lookup = 2 HTTP requests)
        first = await gh.fetch_artifact(repo="org/repo", path="eval.csv", ref="main")
        assert first.commit_sha == "b" * 40
        assert http_calls["count"] == 2

        # 2. Second identical fetch hits ToolCache -> zero new HTTP requests
        second = await gh.fetch_artifact(repo="org/repo", path="eval.csv", ref="main")
        assert second == first
        assert http_calls["count"] == 2

        # 3. 404 client error does NOT trip the breaker
        http_calls["mode"] = "404"
        for idx in range(3):
            with pytest.raises(GitHubArtifactError) as err_404:
                await gh.fetch_artifact(
                    repo="org/repo", path=f"missing-{idx}.csv", ref="main"
                )
            assert err_404.value.status_code == 404
        assert breaker.state == CircuitState.CLOSED
        calls_before_500 = http_calls["count"]

        # 4. Repeated 500 server errors trip the breaker (threshold=2)
        http_calls["mode"] = "500"
        for idx in range(2):
            with pytest.raises(GitHubArtifactError) as err_500:
                await gh.fetch_artifact(
                    repo="org/repo", path=f"broken-{idx}.csv", ref="main"
                )
            assert err_500.value.status_code == 500
        assert breaker.state == CircuitState.OPEN
        calls_at_trip = http_calls["count"]
        assert calls_at_trip == calls_before_500 + 2

        # 5. Subsequent request while OPEN returns 503 immediately without touching HTTP
        with pytest.raises(GitHubArtifactError) as err_open:
            await gh.fetch_artifact(repo="org/repo", path="never-sent.csv", ref="main")
        assert err_open.value.status_code == 503
        assert http_calls["count"] == calls_at_trip


def test_llm_config_extra_api_keys_and_fallback_chain() -> None:
    cfg = LLMConfig(
        api_key=SecretStr("key-primary"),
        extra_api_keys=SecretStr(" key-backup-1 , key-backup-2 , key-primary "),
        fallback_api_key=SecretStr("key-provider-fallback"),
        fallback_model="fallback-model-v1",
    )
    assert [k.get_secret_value() for k in cfg.api_key_pool] == [
        "key-primary",
        "key-backup-1",
        "key-backup-2",
    ]

    # Round-robin key_index rotates the primary key across the pool
    r0 = build_chat_model(cfg, key_index=0)
    r1 = build_chat_model(cfg, key_index=1)
    r2 = build_chat_model(cfg, key_index=2)
    assert isinstance(r0, RunnableWithFallbacks)
    assert isinstance(r1, RunnableWithFallbacks)
    assert isinstance(r2, RunnableWithFallbacks)
    assert len(r0.fallbacks) == 3

    p0 = getattr(r0.runnable, "openai_api_key", None)
    p1 = getattr(r1.runnable, "openai_api_key", None)
    p2 = getattr(r2.runnable, "openai_api_key", None)
    assert isinstance(p0, SecretStr) and p0.get_secret_value() == "key-primary"
    assert isinstance(p1, SecretStr) and p1.get_secret_value() == "key-backup-1"
    assert isinstance(p2, SecretStr) and p2.get_secret_value() == "key-backup-2"

    # BYOK override builds an isolated client with custom key and model
    byok = build_chat_model(
        cfg,
        override=LLMOverride(
            api_key=SecretStr("sk-user-custom"),
            model="custom-user-model",
            base_url="https://api.custom.org/v1",
        ),
    )
    assert not isinstance(byok, RunnableWithFallbacks)
    byok_key = getattr(byok, "openai_api_key", None)
    assert isinstance(byok_key, SecretStr) and byok_key.get_secret_value() == "sk-user-custom"
    assert getattr(byok, "model_name", "") == "custom-user-model"

    # ContextAwareChatRunnable dynamically routes agent/default roles when bind_llm_override is active
    parsed = parse_llm_override(
        {
            "llm_override": {
                "api_key": "sk-ctx-user",
                "model": "claude-3-7-sonnet",
                "base_url": "https://openrouter.ai/api/v1",
            }
        }
    )
    assert parsed is not None
    assert parsed.api_key.get_secret_value() == "sk-ctx-user"

    proxy_agent = ContextAwareChatRunnable(r0, config=cfg, role="agent")
    proxy_router = ContextAwareChatRunnable(r0, config=cfg, role="router")
    assert proxy_agent._active_target() is r0

    with bind_llm_override(parsed):
        active_agent = proxy_agent._active_target()
        active_router = proxy_router._active_target()
        assert active_agent is not r0
        assert getattr(active_agent, "model_name", "") == "claude-3-7-sonnet"
        # Router role always stays on platform model for fast control-plane routing
        assert active_router is r0


def test_prompts_preserve_static_prefix_across_steps_for_kv_cache() -> None:
    # 1. Specialist agent prompts keep 100% identical leading prefix across step=1..4
    ed_s1 = build_editor_planner_prompt("tool-a, tool-b", step=1, max_steps=6)
    ed_s2 = build_editor_planner_prompt("tool-a, tool-b", step=2, max_steps=6)
    assert ed_s1[:-20] == ed_s2[:-20]
    assert ed_s1.endswith("Current step: 1/6.")
    assert ed_s2.endswith("Current step: 2/6.")

    lib_s1 = build_library_planner_prompt("search_items", step=1, max_steps=4)
    lib_s3 = build_library_planner_prompt("search_items", step=3, max_steps=4)
    assert lib_s1[:-20] == lib_s3[:-20]

    proj_s1 = build_project_planner_prompt("list_work_items", step=1, max_steps=5)
    proj_s2 = build_project_planner_prompt("list_work_items", step=2, max_steps=5)
    assert proj_s1[:-20] == proj_s2[:-20]

    # 2. Supervisor prompt keeps static prefix identical regardless of volatile request_context
    sup_a = build_supervisor_prompt(
        "- editor_agent: edit latex",
        request_context={"page_id": "p-1", "cursor_line": 12},
    )
    sup_b = build_supervisor_prompt(
        "- editor_agent: edit latex",
        request_context={"page_id": "p-99", "cursor_line": 480},
    )
    rules_anchor = "6. Preserve the user's language in each target instruction."
    prefix_end_a = sup_a.index(rules_anchor) + len(rules_anchor)
    prefix_end_b = sup_b.index(rules_anchor) + len(rules_anchor)
    assert sup_a[:prefix_end_a] == sup_b[:prefix_end_b]


async def test_supervisor_planner_deterministic_router_cache() -> None:
    llm_calls = {"count": 0}

    def _router_fn(_prompt: LanguageModelInput) -> AIMessage:
        llm_calls["count"] += 1
        return AIMessage(
            content='{"mode": "delegate", "targets": [{"agent": "library_agent", "instruction": "find paper"}], "needs_followup": false, "reasoning": "library lookup"}'
        )

    cache = InMemoryToolCache()
    planner = LLMSupervisorPlanner(
        RunnableLambda(_router_fn),
        cache=cache,
        cache_ttl_seconds=300,
    )
    registry = AgentRegistry(
        [
            AgentSpec(
                name="library_agent",
                description="Library manager",
                handler=RunnableLambda(lambda s: s),
            )
        ]
    )
    state: RootState = {
        "messages": [HumanMessage(content="Find Attention paper in my library")],
        "round_count": 1,
        "agent_outputs": [],
    }
    cfg: RunnableConfig = {}

    # 1. First plan invokes router LLM and caches decision
    d1 = await planner.plan(state, registry, cfg)
    assert d1.mode == "delegate"
    assert d1.targets[0].agent == "library_agent"
    assert llm_calls["count"] == 1

    # 2. Second identical plan hits ToolCache -> 0 new LLM calls ($0 cost)
    d2 = await planner.plan(state, registry, cfg)
    assert d2 == d1
    assert llm_calls["count"] == 1
