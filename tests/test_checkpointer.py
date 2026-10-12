"""Unit tests for pooled PostgreSQL Checkpointer and PostgresConfig bounds."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import langgraph.checkpoint.postgres.aio  # noqa: F401 - preload before monkeypatching pool
import pytest
from langgraph.checkpoint.memory import MemorySaver
from psycopg.rows import dict_row
from pydantic import SecretStr, ValidationError

from ai.shared.config.database import PostgresConfig
from ai.shared.persistence.checkpointer import create_checkpointer


def test_postgres_config_defaults_env_and_pool_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("PG_POOL_MIN_SIZE", raising=False)
    monkeypatch.delenv("PG_POOL_MAX_SIZE", raising=False)
    monkeypatch.delenv("PG_POOL_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("PG_SETUP_ON_STARTUP", raising=False)

    defaults = PostgresConfig()
    assert defaults.is_configured is False
    assert defaults.pg_pool_min_size == 2
    assert defaults.pg_pool_max_size == 10
    assert defaults.pg_pool_timeout_seconds == 10.0
    assert defaults.pg_setup_on_startup is True

    monkeypatch.setenv("DATABASE_URL", "postgresql://flux:secret@localhost:5432/flux")
    monkeypatch.setenv("PG_POOL_MIN_SIZE", "4")
    monkeypatch.setenv("PG_POOL_MAX_SIZE", "24")
    monkeypatch.setenv("PG_POOL_TIMEOUT_SECONDS", "15.5")
    monkeypatch.setenv("PG_SETUP_ON_STARTUP", "false")

    loaded = PostgresConfig()
    assert loaded.is_configured is True
    assert loaded.pg_pool_min_size == 4
    assert loaded.pg_pool_max_size == 24
    assert loaded.pg_pool_timeout_seconds == 15.5
    assert loaded.pg_setup_on_startup is False

    with pytest.raises(ValidationError, match="pg_pool_max_size"):
        PostgresConfig(
            database_url=SecretStr("postgresql://flux:secret@localhost:5432/flux"),
            pg_pool_min_size=12,
            pg_pool_max_size=4,
        )

    with pytest.raises(ValidationError, match="DATABASE_URL"):
        PostgresConfig(
            database_url=SecretStr("mysql://localhost:3306/flux"),
        )


async def test_create_checkpointer_falls_back_to_memory_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = PostgresConfig(database_url=SecretStr(""))
    async with create_checkpointer(cfg) as saver:
        assert isinstance(saver, MemorySaver)


@pytest.mark.parametrize("setup_on_startup", [True, False])
async def test_create_checkpointer_uses_bounded_pool_for_postgres(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    setup_on_startup: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    recorded: dict[str, Any] = {
        "pool_kwargs": None,
        "entered": False,
        "exited": False,
        "saver_args": None,
        "setup_calls": 0,
    }

    class _FakePool:
        def __class_getitem__(cls, item: Any) -> type[Self]:
            del item
            return cls

        def __init__(self, **kwargs: Any) -> None:
            recorded["pool_kwargs"] = kwargs

        async def __aenter__(self) -> Self:
            recorded["entered"] = True
            return self

        async def __aexit__(self, *exc: object) -> None:
            del exc
            recorded["exited"] = True

    class _FakeAsyncPostgresSaver:
        def __init__(self, conn: Any, *, serde: Any = None) -> None:
            recorded["saver_args"] = {"conn": conn, "serde": serde}

        async def setup(self) -> None:
            recorded["setup_calls"] += 1

    monkeypatch.setattr("psycopg_pool.AsyncConnectionPool", _FakePool)
    monkeypatch.setattr(
        "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver",
        _FakeAsyncPostgresSaver,
    )

    cfg = PostgresConfig(
        database_url=SecretStr("postgresql://user:pass@localhost:5432/flux_ai"),
        pg_pool_min_size=3,
        pg_pool_max_size=15,
        pg_pool_timeout_seconds=12.0,
        pg_setup_on_startup=setup_on_startup,
    )

    async with create_checkpointer(cfg) as saver:
        assert isinstance(saver, _FakeAsyncPostgresSaver)
        assert recorded["entered"] is True
        assert recorded["exited"] is False

    assert recorded["exited"] is True
    pool_kw = recorded["pool_kwargs"]
    assert pool_kw["conninfo"] == "postgresql://user:pass@localhost:5432/flux_ai"
    assert pool_kw["min_size"] == 3
    assert pool_kw["max_size"] == 15
    assert pool_kw["timeout"] == 12.0
    assert pool_kw["open"] is False
    assert pool_kw["kwargs"] == {
        "autocommit": True,
        "prepare_threshold": 0,
        "row_factory": dict_row,
    }
    assert recorded["setup_calls"] == (1 if setup_on_startup else 0)
