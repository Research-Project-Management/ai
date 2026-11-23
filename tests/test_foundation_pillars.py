"""Comprehensive tests for all 7 community-standard AI service pillars."""

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig, RunnableLambda
from pydantic import SecretStr

from ai.main import create_app
from ai.orchestrator.contracts import AgentOutput, AgentRegistry, AgentSpec, CitationItem
from ai.orchestrator.graph import build_root_graph
from ai.orchestrator.guards import CitationIntegrityGuard, DefaultInputGuard
from ai.orchestrator.state import RootState
from ai.shared.config.database import DatabaseConfig, PostgresConfig, QdrantConfig, RedisConfig
from ai.shared.config.embedding import EmbeddingConfig
from ai.shared.config.llm import LLMConfig
from ai.shared.config.rate_limit import RateLimitConfig
from ai.shared.config.server import AppEnvironment, ServerConfig
from ai.shared.config.settings import Settings
from ai.shared.llm.factory import ChatRunnable
from ai.shared.resilience.tool_runner import execute_tool_safely


class _SlowStreamingChatModel(BaseChatModel):
    tokens: list[str]
    delay_seconds: float = 0.05

    @property
    def _llm_type(self) -> str:
        return "slow-streaming-stub"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="".join(self.tokens)))]
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        del messages, stop, run_manager, kwargs
        for tok in self.tokens:
            yield ChatGenerationChunk(message=AIMessageChunk(content=tok))

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        del messages, stop, kwargs
        for idx, tok in enumerate(self.tokens):
            if idx > 0:
                await asyncio.sleep(self.delay_seconds)
            msg_chunk = AIMessageChunk(
                content=tok,
                usage_metadata={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            )
            chunk = ChatGenerationChunk(message=msg_chunk)
            if run_manager:
                await run_manager.on_llm_new_token(tok, chunk=chunk)
            yield chunk


def _test_settings() -> Settings:
    return Settings(
        server=ServerConfig(
            app_env=AppEnvironment.TEST,
            internal_api_key=SecretStr("test-key"),
        ),
        llm=LLMConfig(api_key=SecretStr("test-llm-key")),
        embedding=EmbeddingConfig(dimensions=64, collection_name="test_knowledge"),
        database=DatabaseConfig(
            postgres=PostgresConfig(database_url=SecretStr("")),
            redis=RedisConfig(redis_url=SecretStr("")),
            qdrant=QdrantConfig(qdrant_url=""),
        ),
        rate_limit=RateLimitConfig(enabled=True, requests_per_window=50),
    )


def _stub_llm(text: str = "ok", *, with_usage: bool = False) -> ChatRunnable:
    def _respond(_input: LanguageModelInput) -> AIMessage:
        if with_usage:
            return AIMessage(
                content=text,
                usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
            )
        return AIMessage(content=text)

    return RunnableLambda(_respond)


def test_stateless_vs_stateful_execution_and_thread_lifecycle() -> None:
    app = create_app(settings=_test_settings(), llm=_stub_llm("reply-ok", with_usage=True))
    headers = {"X-Internal-Key": "test-key"}

    with TestClient(app) as client:
        stateless_res = client.post(
            "/chat/sync",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "one-off query"}],
                "space_id": "s1",
                "chat_id": "ephemeral-1",
                "stateless": True,
            },
        )
        assert stateless_res.status_code == 200
        stateless_body = stateless_res.json()
        assert stateless_body["stateless"] is True
        assert stateless_body["usage"]["input_tokens"] == 12
        assert stateless_body["usage"]["output_tokens"] == 8
        assert stateless_body["usage"]["total_tokens"] == 20
        assert stateless_body["usage"]["latency_ms"] >= 0.0

        state_ephemeral = client.get("/threads/ephemeral-1/state?space_id=s1", headers=headers)
        assert state_ephemeral.status_code == 200
        assert state_ephemeral.json()["exists"] is False
        assert state_ephemeral.json()["status"] == "empty"

        implicit_stateless = client.post(
            "/chat/sync",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "no chat id"}],
                "space_id": "s1",
            },
        )
        assert implicit_stateless.status_code == 200
        assert implicit_stateless.json()["stateless"] is True

        stateful = client.post(
            "/chat/sync",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "persist me"}],
                "space_id": "s1",
                "chat_id": "thread-1",
            },
        )
        assert stateful.status_code == 200
        assert stateful.json()["stateless"] is False

        state_ok = client.get("/threads/thread-1/state?space_id=s1", headers=headers)
        assert state_ok.status_code == 200
        state_data = state_ok.json()
        assert state_data["exists"] is True
        assert state_data["status"] == "idle"
        assert state_data["message_count"] == 2

        state_forbidden = client.get("/threads/thread-1/state?space_id=s2", headers=headers)
        assert state_forbidden.status_code == 404

        del_wrong_space = client.delete("/threads/thread-1?space_id=s2", headers=headers)
        assert del_wrong_space.status_code == 404

        del_ok = client.delete("/threads/thread-1?space_id=s1", headers=headers)
        assert del_ok.status_code == 200
        assert del_ok.json()["deleted"] is True

        after_del = client.get("/threads/thread-1/state?space_id=s1", headers=headers)
        assert after_del.json()["exists"] is False

        for space, cid in (("s1", "t-a"), ("s1", "t-b"), ("s2", "t-c")):
            res = client.post(
                "/chat/sync",
                headers=headers,
                json={
                    "messages": [{"role": "user", "content": f"hi {cid}"}],
                    "space_id": space,
                    "chat_id": cid,
                },
            )
            assert res.status_code == 200

        purge_s1 = client.delete("/spaces/s1/threads", headers=headers)
        assert purge_s1.status_code == 200
        assert purge_s1.json()["deleted_threads"] == 2

        assert (
            client.get("/threads/t-a/state?space_id=s1", headers=headers).json()["exists"] is False
        )
        assert (
            client.get("/threads/t-c/state?space_id=s2", headers=headers).json()["exists"] is True
        )


def test_active_run_cancellation_stops_sse_stream() -> None:
    slow_llm = _SlowStreamingChatModel(
        tokens=["first ", "second ", "third ", "fourth"], delay_seconds=0.08
    )
    app = create_app(settings=_test_settings(), llm=slow_llm)
    headers = {"X-Internal-Key": "test-key"}

    observed_events: list[tuple[str, dict[str, Any]]] = []
    with (
        TestClient(app) as client,
        client.stream(
            "POST",
            "/chat",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "Stream slowly"}],
                "space_id": "s1",
                "chat_id": "cancel-chat",
            },
        ) as response,
    ):
        assert response.status_code == 200
        current_event = ""
        cancel_sent = False
        for line in response.iter_lines():
            if line.startswith("event: "):
                current_event = line[7:].strip()
            elif line.startswith("data: ") and current_event:
                data = json.loads(line[6:].strip())
                observed_events.append((current_event, data))
                if current_event == "text.delta" and not cancel_sent:
                    cancel_res = client.post(
                        "/threads/cancel-chat/cancel?space_id=s1",
                        headers=headers,
                    )
                    assert cancel_res.status_code == 200
                    assert cancel_res.json()["status"] == "cancelling"
                    cancel_sent = True

    event_names = [name for name, _ in observed_events]
    assert "run.cancelled" in event_names
    assert "run.finished" not in event_names


def test_long_term_store_crud_search_and_space_isolation() -> None:
    app = create_app(settings=_test_settings(), llm=_stub_llm())
    headers = {"X-Internal-Key": "test-key"}

    with TestClient(app) as client:
        put1 = client.put(
            "/store/items",
            headers=headers,
            json={
                "space_id": "s1",
                "namespace": "prefs",
                "key": "writing_style",
                "value": {"tone": "concise", "lang": "vi"},
            },
        )
        assert put1.status_code == 200
        assert put1.json()["value"] == {"tone": "concise", "lang": "vi"}

        put2 = client.put(
            "/store/items",
            headers=headers,
            json={
                "space_id": "s1",
                "namespace": "facts",
                "key": "domain",
                "value": {"topic": "physics"},
            },
        )
        assert put2.status_code == 200

        put_s2 = client.put(
            "/store/items",
            headers=headers,
            json={
                "space_id": "s2",
                "namespace": "prefs",
                "key": "writing_style",
                "value": {"tone": "formal", "lang": "en"},
            },
        )
        assert put_s2.status_code == 200

        get_ok = client.get(
            "/store/items/writing_style?space_id=s1&namespace=prefs", headers=headers
        )
        assert get_ok.status_code == 200
        assert get_ok.json()["value"]["tone"] == "concise"

        get_wrong_ns = client.get(
            "/store/items/writing_style?space_id=s2&namespace=facts",
            headers=headers,
        )
        assert get_wrong_ns.status_code == 404

        search_res = client.post(
            "/store/search",
            headers=headers,
            json={"space_id": "s1", "namespace": "prefs", "filter": {"lang": "vi"}},
        )
        assert search_res.status_code == 200
        assert [item["key"] for item in search_res.json()["items"]] == ["writing_style"]

        del_one = client.delete(
            "/store/items/writing_style?space_id=s1&namespace=prefs",
            headers=headers,
        )
        assert del_one.status_code == 200
        assert (
            client.get(
                "/store/items/writing_style?space_id=s1&namespace=prefs",
                headers=headers,
            ).status_code
            == 404
        )

        purge_s1 = client.delete("/spaces/s1/store", headers=headers)
        assert purge_s1.status_code == 200
        assert purge_s1.json()["deleted_items"] == 1

        still_in_s2 = client.get(
            "/store/items/writing_style?space_id=s2&namespace=prefs",
            headers=headers,
        )
        assert still_in_s2.status_code == 200
        assert still_in_s2.json()["value"]["tone"] == "formal"


def test_qdrant_hybrid_knowledge_upsert_idempotency_search_and_delete() -> None:
    app = create_app(settings=_test_settings(), llm=_stub_llm())
    headers = {"X-Internal-Key": "test-key"}

    with TestClient(app) as client:
        up_v1 = client.post(
            "/knowledge/upsert",
            headers=headers,
            json={
                "space_id": "s1",
                "source_type": "paper",
                "source_id": "p1",
                "title": "Graph Systems",
                "chunks": [
                    {"chunk_index": 0, "content": "quantum attention architecture", "page": 1},
                    {"chunk_index": 1, "content": "multi agent graph routing", "page": 2},
                    {"chunk_index": 2, "content": "obsolete legacy section alpha", "page": 3},
                ],
            },
        )
        assert up_v1.status_code == 200
        assert up_v1.json()["chunks_indexed"] == 3

        up_v2 = client.post(
            "/knowledge/upsert",
            headers=headers,
            json={
                "space_id": "s1",
                "source_type": "paper",
                "source_id": "p1",
                "title": "Graph Systems v2",
                "chunks": [
                    {"chunk_index": 0, "content": "quantum attention updated v2", "page": 1},
                ],
            },
        )
        assert up_v2.status_code == 200
        assert up_v2.json()["chunks_indexed"] == 1

        up_s2 = client.post(
            "/knowledge/upsert",
            headers=headers,
            json={
                "space_id": "s2",
                "source_type": "page",
                "source_id": "pg-secret",
                "title": "Private Notes",
                "content": "confidential space two quantum roadmap",
            },
        )
        assert up_s2.status_code == 200

        search_s1 = client.post(
            "/knowledge/search",
            headers=headers,
            json={"space_id": "s1", "query": "quantum attention", "limit": 10},
        )
        assert search_s1.status_code == 200
        hits_s1 = search_s1.json()["hits"]
        assert len(hits_s1) == 1
        assert hits_s1[0]["source_id"] == "p1"
        assert "obsolete legacy section alpha" not in hits_s1[0]["content"]
        assert all(h["source_id"] != "pg-secret" for h in hits_s1)

        search_wrong_type = client.post(
            "/knowledge/search",
            headers=headers,
            json={
                "space_id": "s1",
                "query": "quantum",
                "source_types": ["page"],
            },
        )
        assert search_wrong_type.status_code == 200
        assert search_wrong_type.json()["hits"] == []

        del_src = client.delete("/knowledge/paper/p1?space_id=s1", headers=headers)
        assert del_src.status_code == 200
        after_src_del = client.post(
            "/knowledge/search",
            headers=headers,
            json={"space_id": "s1", "query": "quantum"},
        )
        assert after_src_del.json()["hits"] == []

        del_s2 = client.delete("/spaces/s2/knowledge", headers=headers)
        assert del_s2.status_code == 200
        after_s2_del = client.post(
            "/knowledge/search",
            headers=headers,
            json={"space_id": "s2", "query": "confidential roadmap"},
        )
        assert after_s2_del.json()["hits"] == []


async def test_agents_discovery_specialist_timeout_and_tool_runner() -> None:
    async def slow_handler(state: RootState, config: RunnableConfig) -> AgentOutput:
        del state, config
        await asyncio.sleep(0.3)
        return AgentOutput(agent="slow_rag", content="never reached")

    registry = AgentRegistry(
        [
            AgentSpec(
                name="slow_rag",
                description="Hybrid document researcher",
                handler=slow_handler,
                capabilities=("hybrid_search", "citations"),
                timeout_seconds=0.05,
            )
        ]
    )

    app = create_app(settings=_test_settings(), llm=_stub_llm("recovered"), registry=registry)
    headers = {"X-Internal-Key": "test-key"}
    with TestClient(app) as client:
        res = client.get("/agents", headers=headers)
    assert res.status_code == 200
    assert res.json()["agents"] == [
        {
            "name": "slow_rag",
            "description": "Hybrid document researcher",
            "capabilities": ["hybrid_search", "citations"],
            "timeout_seconds": 0.05,
        }
    ]

    graph = build_root_graph(llm=_stub_llm("fallback ok"), registry=registry)
    out = await graph.ainvoke(
        {"messages": [HumanMessage(content="Find it")], "intent_hint": "slow_rag"}
    )
    assert out["agent_outputs"][0].status == "error"
    assert "TimeoutError" in (out["agent_outputs"][0].error or "")
    assert out["final_response"] == "fallback ok"

    attempts = 0

    async def flaky_tool() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("transient")
        return "tool-ok"

    tool_result = await execute_tool_safely(flaky_tool, timeout_seconds=1.0, max_retries=1)
    assert tool_result == "tool-ok"
    assert attempts == 2


def test_input_guard_citation_guard_token_usage_and_feedback() -> None:
    async def citing_handler(state: RootState, config: RunnableConfig) -> AgentOutput:
        del state, config
        dup = CitationItem(source_id="doc-1", title="Spec", page=2, snippet="key clause")
        return AgentOutput(
            agent="researcher",
            content="Verified result",
            citations=(dup, dup, CitationItem(source_id="doc-2", title="Notes")),
            is_final=True,
        )

    registry = AgentRegistry(
        [AgentSpec(name="researcher", description="Cited worker", handler=citing_handler)]
    )
    router_stub = RunnableLambda(
        lambda _x: AIMessage(
            content='{"mode": "delegate", "targets": [{"agent": "researcher", "instruction": "cite"}]}'
        )
    )

    app = create_app(
        settings=_test_settings(),
        llm=_stub_llm("unused"),
        router_llm=router_stub,
        registry=registry,
        input_guards=(DefaultInputGuard(max_message_chars=25),),
        output_guards=(CitationIntegrityGuard(),),
    )
    headers = {"X-Internal-Key": "test-key", "X-Trace-Id": "tr-guard-1"}

    with TestClient(app) as client:
        rejected = client.post(
            "/chat/sync",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "x" * 40}],
                "space_id": "s1",
                "chat_id": "c-guard",
            },
        )
        assert rejected.status_code == 400
        assert rejected.json()["detail"]["error"] == "input_guard_rejected"

        cited = client.post(
            "/chat/sync",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "check cite"}],
                "space_id": "s1",
                "chat_id": "c-guard",
            },
        )
        assert cited.status_code == 200
        assert len(cited.json()["citations"]) == 2
        assert [c["source_id"] for c in cited.json()["citations"]] == ["doc-1", "doc-2"]

        fb_post = client.post(
            "/runs/tr-guard-1/feedback",
            headers=headers,
            json={
                "space_id": "s1",
                "score": 1.0,
                "comment": "accurate citations",
                "category": "accuracy",
            },
        )
        assert fb_post.status_code == 201
        assert fb_post.json()["trace_id"] == "tr-guard-1"

        fb_s1 = client.get("/runs/tr-guard-1/feedback?space_id=s1", headers=headers)
        assert fb_s1.status_code == 200
        assert len(fb_s1.json()["items"]) == 1

        fb_s2 = client.get("/runs/tr-guard-1/feedback?space_id=s2", headers=headers)
        assert fb_s2.status_code == 200
        assert fb_s2.json()["items"] == []

    sse_app = create_app(
        settings=_test_settings(),
        llm=_SlowStreamingChatModel(tokens=["hel", "lo"], delay_seconds=0.0),
    )
    with TestClient(sse_app) as client:
        sse_res = client.post(
            "/chat",
            headers={"X-Internal-Key": "test-key", "X-Trace-Id": "tr-sse-usage"},
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "space_id": "s1",
                "stateless": True,
            },
        )
    assert sse_res.status_code == 200
    assert "event: run.finished" in sse_res.text
    for block in sse_res.text.strip().split("\n\n"):
        if "event: run.finished" in block:
            for line in block.splitlines():
                if line.startswith("data: "):
                    payload = json.loads(line[6:].strip())
                    assert payload["usage"]["input_tokens"] > 0
                    assert payload["usage"]["output_tokens"] > 0
                    assert payload["usage"]["latency_ms"] >= 0.0
