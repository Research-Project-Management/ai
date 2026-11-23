"""Tests for SOLID config design: immutability, secret masking, validation, fail-fast."""

from collections.abc import Sequence

import pytest
from pydantic import SecretStr, ValidationError

from ai.shared.config.database import DatabaseConfig, PostgresConfig, QdrantConfig, RedisConfig
from ai.shared.config.embedding import EmbeddingConfig
from ai.shared.config.llm import LLMConfig
from ai.shared.config.rate_limit import RateLimitConfig
from ai.shared.config.server import AppEnvironment, ServerConfig
from ai.shared.config.settings import Settings
from ai.shared.llm.embeddings import (
    DeterministicHashEmbedder,
    FallbackEmbedder,
    OpenAICompatibleEmbedder,
    build_embedder,
)


def test_config_is_immutable() -> None:
    settings = Settings(
        server=ServerConfig(),
        llm=LLMConfig(),
        database=DatabaseConfig(),
    )
    with pytest.raises(ValidationError):
        settings.server.port = 9999  # type: ignore[misc]
    with pytest.raises(ValidationError):
        settings.llm.model = "other"  # type: ignore[misc]


def test_secrets_are_masked_in_repr_and_str() -> None:
    settings = Settings(
        server=ServerConfig(internal_api_key=SecretStr("super-secret-internal")),
        llm=LLMConfig(
            api_key=SecretStr("sk-secret-primary"),
            fallback_api_key=SecretStr("sk-secret-fallback"),
        ),
        database=DatabaseConfig(
            postgres=PostgresConfig(
                database_url=SecretStr("postgresql://user:db-secret-pass@localhost:24781/ai_db")
            ),
            redis=RedisConfig(redis_url=SecretStr("redis://:redis-secret@localhost:31942")),
            qdrant=QdrantConfig(qdrant_api_key=SecretStr("qdrant-secret-key")),
        ),
    )
    rendered = f"{settings!r} {settings!s}"
    for secret in (
        "super-secret-internal",
        "sk-secret-primary",
        "sk-secret-fallback",
        "db-secret-pass",
        "redis-secret",
        "qdrant-secret-key",
    ):
        assert secret not in rendered


def test_domain_validators_reject_invalid_values() -> None:
    with pytest.raises(ValidationError):
        ServerConfig(port=80)

    with pytest.raises(ValidationError):
        LLMConfig(base_url="ftp://invalid-scheme")

    with pytest.raises(ValidationError):
        PostgresConfig(database_url=SecretStr("mysql://user:pass@localhost:3306/db"))

    with pytest.raises(ValidationError):
        RedisConfig(redis_url=SecretStr("http://localhost:31942"))

    with pytest.raises(ValidationError):
        QdrantConfig(qdrant_url="tcp://localhost:27519")


def test_production_mode_fails_fast_when_critical_settings_missing() -> None:
    with pytest.raises(ValidationError):
        ServerConfig(app_env=AppEnvironment.PRODUCTION, internal_api_key=SecretStr(""))

    with pytest.raises(ValidationError):
        Settings(
            server=ServerConfig(
                app_env=AppEnvironment.PRODUCTION,
                internal_api_key=SecretStr("prod-key"),
            ),
            llm=LLMConfig(api_key=SecretStr("")),
            database=DatabaseConfig(
                postgres=PostgresConfig(
                    database_url=SecretStr("postgresql://u:p@ai-postgres:5432/ai_db")
                )
            ),
        )

    with pytest.raises(ValidationError):
        Settings(
            server=ServerConfig(
                app_env=AppEnvironment.PRODUCTION,
                internal_api_key=SecretStr("prod-key"),
            ),
            llm=LLMConfig(api_key=SecretStr("sk-prod")),
            database=DatabaseConfig(postgres=PostgresConfig(database_url=SecretStr(""))),
        )

    assert RedisConfig(redis_url=SecretStr("")).is_configured is False
    assert QdrantConfig(qdrant_url="").is_configured is False

    valid_prod = Settings(
        server=ServerConfig(
            app_env=AppEnvironment.PRODUCTION,
            internal_api_key=SecretStr("prod-key"),
        ),
        llm=LLMConfig(api_key=SecretStr("sk-prod")),
        database=DatabaseConfig(
            postgres=PostgresConfig(
                database_url=SecretStr("postgresql://u:p@ai-postgres:5432/ai_db")
            ),
            redis=RedisConfig(redis_url=SecretStr("redis://ai-redis:6379")),
        ),
    )
    assert valid_prod.server.is_production is True
    assert valid_prod.database.postgres.is_configured is True
    assert valid_prod.database.redis.is_configured is True

    with pytest.raises(ValidationError):
        Settings(
            server=ServerConfig(
                app_env=AppEnvironment.PRODUCTION,
                internal_api_key=SecretStr("prod-key"),
            ),
            llm=LLMConfig(api_key=SecretStr("sk-prod")),
            database=DatabaseConfig(
                postgres=PostgresConfig(
                    database_url=SecretStr("postgresql://u:p@ai-postgres:5432/ai_db")
                ),
                redis=RedisConfig(redis_url=SecretStr("")),
            ),
            rate_limit=RateLimitConfig(enabled=True),
        )

    prod_no_rl = Settings(
        server=ServerConfig(
            app_env=AppEnvironment.PRODUCTION,
            internal_api_key=SecretStr("prod-key"),
        ),
        llm=LLMConfig(api_key=SecretStr("sk-prod")),
        database=DatabaseConfig(
            postgres=PostgresConfig(
                database_url=SecretStr("postgresql://u:p@ai-postgres:5432/ai_db")
            ),
            redis=RedisConfig(redis_url=SecretStr("")),
        ),
        rate_limit=RateLimitConfig(enabled=False),
    )
    assert prod_no_rl.server.is_production is True


def test_llm_role_model_resolution() -> None:
    default_cfg = LLMConfig(model="tozpie-1.0-flash", router_model="", agent_model="")
    assert default_cfg.effective_router_model == "tozpie-1.0-flash"
    assert default_cfg.effective_agent_model == "tozpie-1.0-flash"

    tiered_cfg = LLMConfig(
        model="tozpie-1.0-flash",
        router_model="tozpie-1.0-lite",
        agent_model="tozpie-1.0-pro",
    )
    assert tiered_cfg.effective_router_model == "tozpie-1.0-lite"
    assert tiered_cfg.effective_agent_model == "tozpie-1.0-pro"

    # Per-role base URL & API key isolation + fallback to primary when unset
    fallback_role_cfg = LLMConfig(
        base_url="https://primary.example/v1",
        api_key=SecretStr("sk-primary"),
        extra_api_keys=SecretStr("sk-extra"),
        router_base_url="",
        router_api_key=SecretStr(""),
        agent_base_url="",
        agent_api_key=SecretStr(""),
    )
    assert fallback_role_cfg.effective_base_url("default") == "https://primary.example/v1"
    assert fallback_role_cfg.effective_base_url("router") == "https://primary.example/v1"
    assert fallback_role_cfg.effective_base_url("agent") == "https://primary.example/v1"
    assert [k.get_secret_value() for k in fallback_role_cfg.role_api_key_pool("router")] == [
        "sk-primary",
        "sk-extra",
    ]
    assert [k.get_secret_value() for k in fallback_role_cfg.role_api_key_pool("agent")] == [
        "sk-primary",
        "sk-extra",
    ]

    isolated_role_cfg = LLMConfig(
        base_url="https://primary.example/v1",
        api_key=SecretStr("sk-default-role"),
        router_base_url="https://router.example/v1",
        router_api_key=SecretStr("sk-router-secret"),
        agent_base_url="https://agent.example/v1",
        agent_api_key=SecretStr("sk-agent-secret"),
    )
    assert isolated_role_cfg.effective_base_url("default") == "https://primary.example/v1"
    assert isolated_role_cfg.effective_base_url("router") == "https://router.example/v1"
    assert isolated_role_cfg.effective_base_url("agent") == "https://agent.example/v1"
    assert [k.get_secret_value() for k in isolated_role_cfg.role_api_key_pool("default")] == [
        "sk-default-role"
    ]
    assert [k.get_secret_value() for k in isolated_role_cfg.role_api_key_pool("router")] == [
        "sk-router-secret"
    ]
    assert [k.get_secret_value() for k in isolated_role_cfg.role_api_key_pool("agent")] == [
        "sk-agent-secret"
    ]
    rendered_isolated = f"{isolated_role_cfg!r} {isolated_role_cfg!s}"
    assert "sk-router-secret" not in rendered_isolated
    assert "sk-agent-secret" not in rendered_isolated

    with pytest.raises(ValidationError):
        LLMConfig(router_base_url="ftp://invalid-router")
    with pytest.raises(ValidationError):
        LLMConfig(agent_base_url="ftp://invalid-agent")

    local_emb = build_embedder(
        EmbeddingConfig(provider="local_hash", dimensions=32),
        LLMConfig(api_key=SecretStr("sk-test")),
        allow_remote=True,
    )
    assert isinstance(local_emb, DeterministicHashEmbedder)
    assert local_emb.dimensions == 32


def test_embedding_config_fallback_and_alias_security() -> None:
    cfg = EmbeddingConfig(
        provider="auto",
        base_url="http://127.0.0.1:11434/v1",
        api_key=SecretStr(""),
        model="bge-m3",
        dimensions=1024,
        collection_name="kb_bge_m3",
        collection_alias="kb_active",
        fallback_base_url="https://api.example.com/v1",
        fallback_api_key=SecretStr("sk-emb-fallback-secret"),
    )
    assert "sk-emb-fallback-secret" not in f"{cfg!r} {cfg!s}"
    assert cfg.effective_fallback_model == "bge-m3"
    assert cfg.effective_target_name == "kb_active"
    assert cfg.should_send_dimensions() is False
    assert cfg.should_send_dimensions("text-embedding-3-small") is True
    assert cfg.effective_api_key(LLMConfig(api_key=SecretStr("sk-llm-only"))) == ""

    with pytest.raises(ValidationError):
        EmbeddingConfig(fallback_base_url="ftp://invalid-emb")

    emb = build_embedder(cfg, LLMConfig(api_key=SecretStr("sk-llm-only")), allow_remote=True)
    assert isinstance(emb, FallbackEmbedder)
    assert emb.dimensions == 1024


async def test_fallback_embedder_failover_and_dimension_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingEmbedder:
        dimensions = 16

        async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            raise RuntimeError("local TEI overloaded")

        async def embed_query(self, text: str) -> list[float]:
            raise RuntimeError("local TEI overloaded")

    class _MismatchEmbedder:
        dimensions = 32

        async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            return []

        async def embed_query(self, text: str) -> list[float]:
            return []

    backup = DeterministicHashEmbedder(dimensions=16)
    with pytest.raises(ValueError, match="dimensions"):
        FallbackEmbedder(_FailingEmbedder(), _MismatchEmbedder())

    chained = FallbackEmbedder(_FailingEmbedder(), backup)
    vecs = await chained.embed_documents(["hello vector"])
    qvec = await chained.embed_query("hello vector")
    assert len(vecs) == 1
    assert len(vecs[0]) == 16
    assert len(qvec) == 16

    remote_primary = OpenAICompatibleEmbedder(
        api_key="",
        model="bge-m3",
        dimensions=1024,
        base_url="http://127.0.0.1:11434/v1",
        batch_size=8,
        max_input_chars=10,
    )
    assert remote_primary.dimensions == 1024
    assert remote_primary._client.chunk_size == 8

    captured_docs: list[list[str]] = []
    captured_queries: list[str] = []

    class _StubEmbeddingsClient:
        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            captured_docs.append(texts)
            return [[0.1] * 1024 for _ in texts]

        async def aembed_query(self, text: str) -> list[float]:
            captured_queries.append(text)
            return [0.1] * 1024

    monkeypatch.setattr(remote_primary, "_client", _StubEmbeddingsClient())
    await remote_primary.embed_documents(["0123456789ABCDEF", "short"])
    await remote_primary.embed_query("query-0123456789")
    assert captured_docs == [["0123456789", "short"]]
    assert captured_queries == ["query-0123"]

    assert EmbeddingConfig.model_fields["timeout_seconds"].default == 60.0
    assert EmbeddingConfig.model_fields["batch_size"].default == 16
    assert EmbeddingConfig.model_fields["max_input_chars"].default == 0
    with pytest.raises(ValidationError):
        EmbeddingConfig(batch_size=0)
    with pytest.raises(ValidationError):
        EmbeddingConfig(max_input_chars=-1)
