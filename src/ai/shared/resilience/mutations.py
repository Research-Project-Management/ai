"""Per-turn Saga Mutation Journal and LIFO compensation runner for safe AI undo/revert."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict

from ai.shared.gateway.transport import BackendTransport
from ai.shared.resilience.journal_store import (
    JournalStore,
    MemoryJournal,
    MutationRecord,
    RedisJournal,
    build_journal,
)


class MutationRevertError(Exception):
    """HTTP error raised while executing a compensating mutation revert request."""

    def __init__(self, status_code: int | None, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class RevertStepResult(BaseModel):
    """Outcome of executing a single compensating HTTP action during turn revert."""

    model_config = ConfigDict(frozen=True)

    tool: str
    entity_id: str
    label: str = ""
    status: Literal["reverted", "already_gone", "failed"]
    error: str = ""


_TURN_JOURNAL_CTX: ContextVar[tuple[JournalStore, str, str] | None] = ContextVar(
    "flux_turn_mutation_journal", default=None
)


@contextmanager
def bind_turn_journal(
    journal: JournalStore | None,
    *,
    space_id: str,
    trace_id: str,
) -> Iterator[None]:
    """Bind the active `(journal, space_id, trace_id)` tuple to the current async execution."""
    if journal is None or not space_id.strip() or not trace_id.strip():
        yield
        return
    token = _TURN_JOURNAL_CTX.set((journal, space_id, trace_id))
    try:
        yield
    finally:
        _TURN_JOURNAL_CTX.reset(token)


def is_turn_journal_bound() -> bool:
    """Return True when an active turn mutation journal is bound in the current context."""
    return _TURN_JOURNAL_CTX.get() is not None


def record_turn_mutation(mutation: MutationRecord) -> None:
    """Append a mutation record to the active turn journal when bound (safe no-op otherwise)."""
    bound = _TURN_JOURNAL_CTX.get()
    if bound is None:
        return
    journal, space_id, trace_id = bound
    journal.record(space_id, trace_id, mutation)


def record_patch_undo(
    *,
    tool: str,
    entity_type: str,
    entity_id: str,
    label: str,
    path: str,
    before_state: dict[str, Any],
    method: Literal["PATCH", "PUT"] = "PATCH",
) -> None:
    """Record a snapshot-based `PATCH`/`PUT` rollback action into the active turn journal."""
    if not entity_id.strip() or not before_state:
        return
    record_turn_mutation(
        MutationRecord(
            tool=tool,
            entity_type=entity_type,
            entity_id=entity_id.strip(),
            label=label,
            undo_method=method,
            undo_path=path,
            undo_body=dict(before_state),
        )
    )


def build_revert_artifact(
    journal: JournalStore | None,
    *,
    space_id: str,
    trace_id: str,
) -> dict[str, Any] | None:
    """Serialize any mutations recorded for `(space_id, trace_id)` as a `revertible_turn` artifact."""
    if journal is None or not space_id.strip() or not trace_id.strip():
        return None
    mutations = journal.list_mutations(space_id, trace_id)
    if not mutations:
        return None
    return {
        "trace_id": trace_id,
        "count": len(mutations),
        "items": [m.model_dump(mode="json") for m in mutations],
    }


async def revert_turn_mutations(
    journal: JournalStore,
    *,
    space_id: str,
    trace_id: str,
    transport: BackendTransport,
    gateway_error: type[Exception] = MutationRevertError,
    entity_ids: Sequence[str] | None = None,
) -> list[RevertStepResult]:
    """Undo a turn's recorded (or selected) mutations in LIFO order with idempotent 404 handling."""
    mutations = await journal.alist(space_id, trace_id)
    if not mutations:
        return []

    selected = {eid.strip() for eid in entity_ids if eid and eid.strip()} if entity_ids else None
    targets = [m for m in mutations if selected is None or m.entity_id in selected]
    if not targets:
        return []

    results: list[RevertStepResult] = []
    for mut in reversed(targets):
        try:
            await transport.request(mut.undo_method, mut.undo_path, json_body=mut.undo_body)
            results.append(
                RevertStepResult(
                    tool=mut.tool,
                    entity_id=mut.entity_id,
                    label=mut.label,
                    status="reverted",
                )
            )
        except (gateway_error, httpx.HTTPError, TimeoutError, OSError) as exc:
            if getattr(exc, "status_code", None) == 404:
                results.append(
                    RevertStepResult(
                        tool=mut.tool,
                        entity_id=mut.entity_id,
                        label=mut.label,
                        status="already_gone",
                    )
                )
            else:
                results.append(
                    RevertStepResult(
                        tool=mut.tool,
                        entity_id=mut.entity_id,
                        label=mut.label,
                        status="failed",
                        error=str(exc)[:200],
                    )
                )

    if selected is None:
        if not any(item.status == "failed" for item in results):
            await journal.aclear(space_id, trace_id)
    else:
        done_ids = [
            item.entity_id for item in results if item.status in ("reverted", "already_gone")
        ]
        if done_ids:
            await journal.aremove(space_id, trace_id, done_ids)
    return results


__all__ = [
    "JournalStore",
    "MemoryJournal",
    "MutationRecord",
    "MutationRevertError",
    "RedisJournal",
    "RevertStepResult",
    "bind_turn_journal",
    "build_journal",
    "build_revert_artifact",
    "is_turn_journal_bound",
    "record_patch_undo",
    "record_turn_mutation",
    "revert_turn_mutations",
]
