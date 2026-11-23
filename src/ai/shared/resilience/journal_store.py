"""Repository & Factory for per-turn mutation history (Memory & Redis write-through)."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError
from redis.asyncio import Redis
from redis.exceptions import RedisError

from ai.shared.persistence.redis import await_redis

logger = logging.getLogger(__name__)


class MutationRecord(BaseModel):
    """Immutable descriptor of a completed write mutation and its inverse Backend HTTP action."""

    model_config = ConfigDict(frozen=True)

    tool: str
    entity_type: str
    entity_id: str
    label: str = ""
    undo_method: Literal["DELETE", "PATCH", "PUT", "POST"]
    undo_path: str
    undo_body: dict[str, Any] | None = None


class JournalStore(Protocol):
    """Repository contract for bounded, space-scoped per-turn mutation logs."""

    def record(self, space_id: str, trace_id: str, item: MutationRecord) -> None: ...

    def list_mutations(self, space_id: str, trace_id: str) -> tuple[MutationRecord, ...]: ...

    def remove(self, space_id: str, trace_id: str, entity_ids: Sequence[str]) -> None: ...

    def clear(self, space_id: str, trace_id: str) -> None: ...

    async def alist(self, space_id: str, trace_id: str) -> tuple[MutationRecord, ...]: ...

    async def aremove(self, space_id: str, trace_id: str, entity_ids: Sequence[str]) -> None: ...

    async def aclear(self, space_id: str, trace_id: str) -> None: ...


class MemoryJournal:
    """Bounded LRU in-memory mutation repository evicting oldest turns past `max_turns`."""

    def __init__(self, *, max_turns: int = 512) -> None:
        self._max_turns = max(1, max_turns)
        self._turns: OrderedDict[tuple[str, str], list[MutationRecord]] = OrderedDict()

    def record(self, space_id: str, trace_id: str, item: MutationRecord) -> None:
        key = (space_id.strip(), trace_id.strip())
        if not key[0] or not key[1]:
            return
        bucket = self._turns.get(key)
        if bucket is not None:
            bucket.append(item)
            self._turns.move_to_end(key)
            return
        while len(self._turns) >= self._max_turns:
            self._turns.popitem(last=False)
        self._turns[key] = [item]

    def replace(self, space_id: str, trace_id: str, items: Sequence[MutationRecord]) -> None:
        """Replace cached items for `(space_id, trace_id)` in local LRU order."""
        key = (space_id.strip(), trace_id.strip())
        if not key[0] or not key[1]:
            return
        if not items:
            self._turns.pop(key, None)
            return
        while key not in self._turns and len(self._turns) >= self._max_turns:
            self._turns.popitem(last=False)
        self._turns[key] = list(items)
        self._turns.move_to_end(key)

    def list_mutations(self, space_id: str, trace_id: str) -> tuple[MutationRecord, ...]:
        key = (space_id.strip(), trace_id.strip())
        bucket = self._turns.get(key)
        return tuple(bucket) if bucket else ()

    def remove(self, space_id: str, trace_id: str, entity_ids: Sequence[str]) -> None:
        key = (space_id.strip(), trace_id.strip())
        targets = {eid.strip() for eid in entity_ids if eid and eid.strip()}
        if not targets:
            return
        bucket = self._turns.get(key)
        if not bucket:
            return
        remaining = [m for m in bucket if m.entity_id not in targets]
        if remaining:
            self._turns[key] = remaining
        else:
            self._turns.pop(key, None)

    def clear(self, space_id: str, trace_id: str) -> None:
        key = (space_id.strip(), trace_id.strip())
        self._turns.pop(key, None)

    async def alist(self, space_id: str, trace_id: str) -> tuple[MutationRecord, ...]:
        return self.list_mutations(space_id, trace_id)

    async def aremove(self, space_id: str, trace_id: str, entity_ids: Sequence[str]) -> None:
        self.remove(space_id, trace_id, entity_ids)

    async def aclear(self, space_id: str, trace_id: str) -> None:
        self.clear(space_id, trace_id)


class RedisJournal:
    """Write-through journal combining fast local cache with Redis persistence."""

    def __init__(
        self,
        redis_client: Redis,
        *,
        max_turns: int = 512,
        ttl_seconds: int = 604_800,
        key_prefix: str = "flux:ai:turn_journal",
    ) -> None:
        self._redis = redis_client
        self._local = MemoryJournal(max_turns=max_turns)
        self._ttl = max(60, ttl_seconds)
        self._prefix = key_prefix
        self._pending: set[asyncio.Task[Any]] = set()

    def _key(self, space_id: str, trace_id: str) -> str:
        return f"{self._prefix}:{space_id.strip()}:{trace_id.strip()}"

    def _spawn(self, coro: Any) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Close unawaited coroutine if called outside an active event loop.
            coro.close()
            return
        task = loop.create_task(coro)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _flush_pending(self) -> None:
        if not self._pending:
            return
        tasks = list(self._pending)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _rpush_item(self, space_id: str, trace_id: str, item: MutationRecord) -> None:
        key = self._key(space_id, trace_id)
        try:
            await await_redis(self._redis.rpush(key, item.model_dump_json()))
            await await_redis(self._redis.expire(key, self._ttl))
        except (RedisError, OSError, TimeoutError) as exc:
            logger.warning("Redis journal write failed: %s", type(exc).__name__)

    async def _write_list(
        self, space_id: str, trace_id: str, items: Sequence[MutationRecord]
    ) -> None:
        key = self._key(space_id, trace_id)
        try:
            await await_redis(self._redis.delete(key))
            if items:
                payloads = [item.model_dump_json() for item in items]
                await await_redis(self._redis.rpush(key, *payloads))
                await await_redis(self._redis.expire(key, self._ttl))
        except (RedisError, OSError, TimeoutError) as exc:
            logger.warning("Redis journal update failed: %s", type(exc).__name__)

    def record(self, space_id: str, trace_id: str, item: MutationRecord) -> None:
        clean_space = space_id.strip()
        clean_trace = trace_id.strip()
        if not clean_space or not clean_trace:
            return
        self._local.record(clean_space, clean_trace, item)
        self._spawn(self._rpush_item(clean_space, clean_trace, item))

    def list_mutations(self, space_id: str, trace_id: str) -> tuple[MutationRecord, ...]:
        return self._local.list_mutations(space_id, trace_id)

    def remove(self, space_id: str, trace_id: str, entity_ids: Sequence[str]) -> None:
        self._local.remove(space_id, trace_id, entity_ids)
        remaining = self._local.list_mutations(space_id, trace_id)
        self._spawn(self._write_list(space_id, trace_id, remaining))

    def clear(self, space_id: str, trace_id: str) -> None:
        self._local.clear(space_id, trace_id)
        self._spawn(self._write_list(space_id, trace_id, ()))

    async def alist(self, space_id: str, trace_id: str) -> tuple[MutationRecord, ...]:
        clean_space = space_id.strip()
        clean_trace = trace_id.strip()
        if not clean_space or not clean_trace:
            return ()
        await self._flush_pending()
        key = self._key(clean_space, clean_trace)
        try:
            raw_items = await await_redis(self._redis.lrange(key, 0, -1))
            if raw_items:
                parsed: list[MutationRecord] = []
                for raw in raw_items:
                    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                    try:
                        parsed.append(MutationRecord.model_validate_json(text))
                    except ValidationError:
                        continue
                if parsed:
                    self._local.replace(clean_space, clean_trace, parsed)
                    return tuple(parsed)
        except (RedisError, OSError, TimeoutError) as exc:
            logger.warning("Redis journal read failed: %s", type(exc).__name__)
        return self._local.list_mutations(clean_space, clean_trace)

    async def aremove(self, space_id: str, trace_id: str, entity_ids: Sequence[str]) -> None:
        clean_space = space_id.strip()
        clean_trace = trace_id.strip()
        targets = {eid.strip() for eid in entity_ids if eid and eid.strip()}
        if not clean_space or not clean_trace or not targets:
            return
        current = await self.alist(clean_space, clean_trace)
        remaining = [m for m in current if m.entity_id not in targets]
        self._local.replace(clean_space, clean_trace, remaining)
        await self._write_list(clean_space, clean_trace, remaining)

    async def aclear(self, space_id: str, trace_id: str) -> None:
        clean_space = space_id.strip()
        clean_trace = trace_id.strip()
        if not clean_space or not clean_trace:
            return
        await self._flush_pending()
        self._local.clear(clean_space, clean_trace)
        await self._write_list(clean_space, clean_trace, ())


def build_journal(
    redis_client: Redis | None = None,
    *,
    max_turns: int = 512,
    ttl_seconds: int = 604_800,
) -> JournalStore:
    """Factory returning Redis-backed write-through journal or in-memory fallback."""
    if redis_client is None:
        return MemoryJournal(max_turns=max_turns)
    return RedisJournal(redis_client, max_turns=max_turns, ttl_seconds=ttl_seconds)


__all__ = [
    "JournalStore",
    "MemoryJournal",
    "MutationRecord",
    "RedisJournal",
    "build_journal",
]
