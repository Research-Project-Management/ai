"""LangGraph long-term cross-thread Store lifecycle (Postgres or in-memory)."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from ai.shared.config.database import PostgresConfig


@asynccontextmanager
async def create_memory_store(config: PostgresConfig) -> AsyncIterator[BaseStore]:
    """Yield an async cross-thread BaseStore using Postgres when configured."""
    if not config.is_configured:
        yield InMemoryStore()
        return

    from langgraph.store.postgres.aio import AsyncPostgresStore

    async with AsyncPostgresStore.from_conn_string(
        config.database_url.get_secret_value()
    ) as store:
        if config.pg_setup_on_startup:
            await store.setup()
        yield store
