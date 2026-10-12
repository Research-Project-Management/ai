"""Async Redis client factory and typing helper."""

import inspect
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from typing import cast

from redis.asyncio import Redis

from ai.shared.config.database import RedisConfig


async def await_redis[T](value: T | Awaitable[T]) -> T:
    """Resolve `redis.asyncio` methods typed as `T | Awaitable[T]`."""
    if inspect.isawaitable(value):
        return await cast(Awaitable[T], value)
    return value


@asynccontextmanager
async def create_redis_client(config: RedisConfig) -> AsyncIterator[Redis | None]:
    """Yield a managed async Redis client, or `None` when `REDIS_URL` is empty."""
    if not config.is_configured:
        yield None
        return

    client: Redis = Redis.from_url(
        config.redis_url.get_secret_value(),
        decode_responses=True,
        socket_connect_timeout=1.5,
        socket_timeout=2.0,
    )
    try:
        yield client
    finally:
        await client.aclose()
