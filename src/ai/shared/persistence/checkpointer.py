"""LangGraph checkpointer lifecycle (pooled Postgres in production, in-memory fallback)."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Final

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from ai.shared.config.database import PostgresConfig

_ALLOWED_MSGPACK_MODULES: Final[tuple[tuple[str, str], ...]] = (
    ("ai.agents.rag.retrieval.contracts", "RankedPassage"),
    ("ai.orchestrator.contracts", "AgentOutput"),
    ("ai.orchestrator.contracts", "AgentTaskAssignment"),
    ("ai.orchestrator.contracts", "CitationItem"),
    ("ai.orchestrator.contracts", "RoutingDecision"),
    ("ai.shared.contracts.run_control", "RunUsage"),
)


@asynccontextmanager
async def create_checkpointer(config: PostgresConfig) -> AsyncIterator[BaseCheckpointSaver]:
    """Yield an async LangGraph checkpointer configured from PostgresConfig."""
    serde = JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)
    if not config.is_configured:
        yield MemorySaver(serde=serde)
        return

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg import AsyncConnection
    from psycopg.rows import DictRow, dict_row
    from psycopg_pool import AsyncConnectionPool

    pool = AsyncConnectionPool(
        conninfo=config.database_url.get_secret_value(),
        connection_class=AsyncConnection[DictRow],
        min_size=config.pg_pool_min_size,
        max_size=config.pg_pool_max_size,
        timeout=config.pg_pool_timeout_seconds,
        open=False,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
    )
    async with pool:
        saver = AsyncPostgresSaver(pool, serde=serde)
        if config.pg_setup_on_startup:
            await saver.setup()
        yield saver
