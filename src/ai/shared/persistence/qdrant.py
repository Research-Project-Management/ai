"""Async Qdrant client factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from qdrant_client import AsyncQdrantClient

from ai.shared.config.database import QdrantConfig


@asynccontextmanager
async def create_qdrant_client(
    config: QdrantConfig,
) -> AsyncIterator[AsyncQdrantClient | None]:
    """Yield a managed async Qdrant client, or `None` when `QDRANT_URL` is empty."""
    if not config.is_configured:
        yield None
        return

    api_key = config.qdrant_api_key.get_secret_value().strip() or None
    client = AsyncQdrantClient(
        url=config.qdrant_url,
        api_key=api_key,
        timeout=10,
    )
    try:
        yield client
    finally:
        await client.close()
