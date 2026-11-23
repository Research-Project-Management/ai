"""LangGraph Time Travel helpers for inspecting checkpoints and rewinding conversation state."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict

from ai.shared.llm.output import extract_text


class CheckpointNotFoundError(Exception):
    """Raised when the requested checkpoint does not exist for a thread."""


class CheckpointSummary(BaseModel):
    """Compact descriptor of one persisted LangGraph checkpoint snapshot."""

    model_config = ConfigDict(frozen=True)

    checkpoint_id: str
    parent_checkpoint_id: str | None = None
    created_at: str | None = None
    step: int | None = None
    message_count: int = 0
    route: str | None = None
    preview: str = ""


def _extract_preview(messages: Any) -> tuple[int, str]:
    if not isinstance(messages, Sequence) or isinstance(messages, str | bytes):
        return 0, ""
    preview = ""
    for msg in reversed(messages):
        is_human = isinstance(msg, HumanMessage) or (
            isinstance(msg, dict) and msg.get("role") == "user"
        )
        if not is_human:
            continue
        raw = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        text = extract_text(raw).strip()
        if text:
            preview = text[:120]
            break
    return len(messages), preview


async def list_thread_checkpoints(
    graph: CompiledStateGraph,
    *,
    chat_id: str,
    limit: int = 20,
) -> list[CheckpointSummary]:
    """Return recent persisted checkpoints for `chat_id` in newest-first order."""
    clean_limit = max(1, min(limit, 100))
    run_config: RunnableConfig = {"configurable": {"thread_id": chat_id}}
    summaries: list[CheckpointSummary] = []

    async for snap in graph.aget_state_history(run_config, limit=clean_limit):
        cfg = snap.config.get("configurable") or {}
        checkpoint_id = str(cfg.get("checkpoint_id") or "").strip()
        if not checkpoint_id:
            continue

        parent_cfg = (snap.parent_config or {}).get("configurable") or {}
        parent_id = str(parent_cfg.get("checkpoint_id") or "").strip() or None
        meta = snap.metadata if isinstance(snap.metadata, dict) else {}
        raw_step = meta.get("step")
        step = int(raw_step) if isinstance(raw_step, int) else None

        values = snap.values if isinstance(snap.values, dict) else {}
        msg_count, preview = _extract_preview(values.get("messages"))
        route = str(values.get("route")) if values.get("route") else None

        summaries.append(
            CheckpointSummary(
                checkpoint_id=checkpoint_id,
                parent_checkpoint_id=parent_id,
                created_at=str(snap.created_at) if snap.created_at is not None else None,
                step=step,
                message_count=msg_count,
                route=route,
                preview=preview,
            )
        )
    return summaries


async def rewind_thread_state(
    graph: CompiledStateGraph,
    *,
    chat_id: str,
    checkpoint_id: str,
    space_id: str | None = None,
) -> tuple[str, int]:
    """Fork/rewind thread head back to `checkpoint_id` and return `(new_checkpoint_id, message_count)`."""
    lookup_cfg: RunnableConfig = {
        "configurable": {
            "thread_id": chat_id,
            "checkpoint_ns": "",
            "checkpoint_id": checkpoint_id,
        }
    }
    snapshot = await graph.aget_state(lookup_cfg)
    if snapshot.created_at is None and not snapshot.values:
        raise CheckpointNotFoundError(
            f"Checkpoint '{checkpoint_id}' not found for thread '{chat_id}'"
        )

    values = snapshot.values if isinstance(snapshot.values, dict) else {}
    msg_count, _ = _extract_preview(values.get("messages"))

    snap_cfg: dict[str, Any] = dict(snapshot.config.get("configurable") or {})
    snap_cfg.setdefault("thread_id", chat_id)
    snap_cfg.setdefault("checkpoint_ns", "")
    snap_cfg["checkpoint_id"] = checkpoint_id
    if space_id:
        snap_cfg["space_id"] = space_id

    update_cfg: RunnableConfig = (
        {"configurable": snap_cfg, "metadata": {"space_id": space_id}}
        if space_id
        else {"configurable": snap_cfg}
    )
    new_cfg = await graph.aupdate_state(update_cfg, values=None)
    new_id = str((new_cfg.get("configurable") or {}).get("checkpoint_id") or "")
    return new_id, msg_count


__all__ = [
    "CheckpointNotFoundError",
    "CheckpointSummary",
    "list_thread_checkpoints",
    "rewind_thread_state",
]
