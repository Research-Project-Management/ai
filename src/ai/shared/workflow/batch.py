"""Shared execution-batch helpers: observation records, terminal-batch detection, citation dedupe."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class HasProvenance(Protocol):
    @property
    def source_id(self) -> str: ...
    @property
    def page(self) -> int | None: ...
    @property
    def snippet(self) -> str: ...


class ToolResult(Protocol):
    """Structural view of a tool result shared across batch helpers and executor."""

    @property
    def tool(self) -> str: ...
    @property
    def status(self) -> str: ...
    @property
    def data(self) -> Any: ...
    @property
    def error(self) -> str | None: ...
    @property
    def citations(self) -> Sequence[Any]: ...


def observation_of(result: ToolResult) -> dict[str, Any]:
    """Serialize a tool result into the compact observation record fed back to the planner."""
    return {
        "tool": result.tool,
        "status": result.status,
        **({"data": result.data} if result.data is not None else {}),
        **({"error": result.error} if result.error else {}),
    }


def dedupe_citations[C: HasProvenance](existing: Sequence[C], new_items: Sequence[C]) -> list[C]:
    """Append `new_items` to `existing`, skipping duplicates by (source, page, snippet)."""
    merged = list(existing)
    if not new_items:
        return merged
    seen = {(c.source_id, c.page, c.snippet) for c in merged}
    for cite in new_items:
        key = (cite.source_id, cite.page, cite.snippet)
        if key not in seen:
            seen.add(key)
            merged.append(cite)
    return merged


class BatchTracker:
    """Decide whether an executed batch is terminal.

    A batch is terminal when every approved call was a non-read tool and all succeeded: the
    planner would only answer `respond` next, so the graph can go straight to synthesis.
    """

    def __init__(self, *, expected_calls: int) -> None:
        self._all_ok = expected_calls > 0
        self._all_mutating = expected_calls > 0

    def observe(self, *, risk_tier: str | None, status: str | None) -> None:
        if risk_tier is None or status is None:
            self._all_ok = False
            return
        if risk_tier == "read":
            self._all_mutating = False
        if status != "ok":
            self._all_ok = False

    @property
    def is_terminal(self) -> bool:
        return self._all_ok and self._all_mutating
