"""Smoke and wiring tests using pure Dependency Injection (no monkeypatching)."""

from fastapi.testclient import TestClient
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr

from ai.main import create_app
from ai.orchestrator.graph import build_root_graph
from ai.shared.config.database import DatabaseConfig, PostgresConfig, QdrantConfig, RedisConfig
from ai.shared.config.llm import LLMConfig
from ai.shared.config.server import AppEnvironment, ServerConfig
from ai.shared.config.settings import Settings
from ai.shared.llm.factory import ChatRunnable


def _stub_llm() -> ChatRunnable:
    def _respond(_messages: LanguageModelInput) -> AIMessage:
        return AIMessage(content="stub-ok")

    return RunnableLambda(_respond)


def _test_settings(*, internal_api_key: str = "test-internal-key") -> Settings:
    return Settings(
        server=ServerConfig(
            app_env=AppEnvironment.TEST,
            internal_api_key=SecretStr(internal_api_key),
        ),
        llm=LLMConfig(api_key=SecretStr("test-llm-key")),
        database=DatabaseConfig(
            postgres=PostgresConfig(database_url=SecretStr("")),
            redis=RedisConfig(redis_url=SecretStr("")),
            qdrant=QdrantConfig(qdrant_url=""),
        ),
    )


def test_health_endpoint() -> None:
    app = create_app(settings=_test_settings(), llm=_stub_llm())
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app_env"] == "test"
    assert [comp["name"] for comp in body["components"]] == [
        "postgres",
        "redis",
        "qdrant",
    ]
    assert all(comp["status"] == "disabled" for comp in body["components"])


def test_chat_sync_enforces_internal_key_and_runs_graph() -> None:
    app = create_app(settings=_test_settings(internal_api_key="secret-key"), llm=_stub_llm())
    payload = {
        "messages": [{"role": "user", "content": "Hello"}],
        "space_id": "s1",
        "chat_id": "c1",
    }
    with TestClient(app) as client:
        unauthorized = client.post("/chat/sync", json=payload)
        assert unauthorized.status_code == 401

        authorized = client.post(
            "/chat/sync",
            json=payload,
            headers={"X-Internal-Key": "secret-key"},
        )
    assert authorized.status_code == 200
    body = authorized.json()
    assert body["content"] == "stub-ok"
    assert body["route"] == "direct"
    assert body["status"] == "completed"


async def test_root_graph_compiles_and_invokes() -> None:
    graph = build_root_graph(llm=_stub_llm())
    result = await graph.ainvoke({"messages": [HumanMessage(content="ping")]})
    assert result["final_response"] == "stub-ok"
