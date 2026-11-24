"""Tests for platform LLM Model Catalog whitelist, `/models` endpoint, and `model_id` routing."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr

from ai.main import create_app
from ai.shared.config.database import DatabaseConfig, PostgresConfig, QdrantConfig, RedisConfig
from ai.shared.config.llm import LLMConfig
from ai.shared.config.server import AppEnvironment, ServerConfig
from ai.shared.config.settings import Settings
from ai.shared.llm.catalog import (
    DEFAULT_CATALOG,
    list_model_catalog,
    resolve_catalog_override,
)
from ai.shared.llm.factory import ChatRunnable, parse_llm_override


def _stub_llm() -> ChatRunnable:
    return RunnableLambda(lambda _messages: AIMessage(content="ok"))


def _test_settings() -> Settings:
    return Settings(
        server=ServerConfig(
            app_env=AppEnvironment.TEST,
            internal_api_key=SecretStr("catalog-test-key"),
        ),
        llm=LLMConfig(
            base_url="https://api.platform.example/v1",
            api_key=SecretStr("sk-platform-secret"),
            model="gemini-2.5-flash",
            router_model="deepseek-v4-flash",
            agent_model="deepseek-v3.2",
        ),
        database=DatabaseConfig(
            postgres=PostgresConfig(database_url=SecretStr("")),
            redis=RedisConfig(redis_url=SecretStr("")),
            qdrant=QdrantConfig(qdrant_url=""),
        ),
    )


def test_models_endpoint_requires_key_and_returns_catalog() -> None:
    app = create_app(settings=_test_settings(), llm=_stub_llm())
    with TestClient(app) as client:
        unauth = client.get("/models")
        assert unauth.status_code == 401

        res = client.get("/models", headers={"X-Internal-Key": "catalog-test-key"})
        assert res.status_code == 200
        models = res.json()["models"]
        assert [m["id"] for m in models] == [entry.id for entry in DEFAULT_CATALOG]
        assert models[0]["id"] == "auto"
        assert models[0]["is_default"] is True
        assert all("api_key" not in m for m in models)


def test_resolve_catalog_override_enforces_whitelist_and_byok_precedence() -> None:
    cfg = _test_settings().llm
    catalog_ids = {m.id for m in list_model_catalog()}
    assert {
        "auto",
        "gemini-2.5-flash",
        "deepseek-v3.2",
        "deepseek-v4-flash",
        "claude-sonnet-4-6",
    } <= catalog_ids

    assert resolve_catalog_override(None, cfg) is None
    assert resolve_catalog_override({}, cfg) is None
    assert resolve_catalog_override({"model_id": "auto"}, cfg) is None
    assert resolve_catalog_override({"model_id": "gpt-4-unlisted-hack"}, cfg) is None

    selected = resolve_catalog_override({"model_id": "deepseek-v3.2"}, cfg)
    assert selected is not None
    assert selected.model == "deepseek-v3.2"
    assert selected.api_key.get_secret_value() == "sk-platform-secret"
    assert selected.base_url == "https://api.platform.example/v1"

    # BYOK `llm_override` takes precedence over catalog `model_id` when both present
    ctx = {
        "model_id": "deepseek-v3.2",
        "llm_override": {
            "api_key": "sk-user-byok",
            "model": "custom-model",
            "base_url": "https://byok.example/v1",
        },
    }
    chosen = parse_llm_override(ctx) or resolve_catalog_override(ctx, cfg)
    assert chosen is not None
    assert chosen.api_key.get_secret_value() == "sk-user-byok"
    assert chosen.model == "custom-model"


def test_catalog_env_override_caches_per_raw_value_and_fails_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom = [
        {
            "id": "custom-fast",
            "label": "Custom Fast",
            "provider": "Custom",
            "tier": "fast",
            "description": "Custom test model",
        }
    ]
    monkeypatch.setenv("LLM_MODEL_CATALOG_JSON", json.dumps(custom))
    assert [m.id for m in list_model_catalog()] == ["custom-fast"]

    monkeypatch.setenv("LLM_MODEL_CATALOG_JSON", "{invalid json")
    assert [m.id for m in list_model_catalog()] == [entry.id for entry in DEFAULT_CATALOG]

    monkeypatch.delenv("LLM_MODEL_CATALOG_JSON", raising=False)
    assert [m.id for m in list_model_catalog()] == [entry.id for entry in DEFAULT_CATALOG]
