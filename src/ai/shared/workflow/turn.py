"""Turn inputs shared by every specialist: the active request text and a short conversation tail."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from ai.shared.llm.output import extract_text

_TAIL_CHARS = 400


def latest_query(state: Mapping[str, Any]) -> str:
    """Return the supervisor's delegated instruction, else the latest user message text."""
    instruction = str(state.get("active_instruction") or "").strip()
    if instruction:
        return instruction
    for msg in reversed(state.get("messages") or ()):
        if isinstance(msg, HumanMessage):
            return extract_text(msg.content).strip()
    return ""


def recent_tail(messages: Sequence[BaseMessage], *, limit: int = 4) -> list[str]:
    """Render the last `limit` messages as `role: text` lines, truncated for prompt budgets."""
    lines: list[str] = []
    for msg in messages[-limit:]:
        role = (
            "user"
            if isinstance(msg, HumanMessage)
            else "assistant"
            if isinstance(msg, AIMessage)
            else "system"
        )
        text = extract_text(msg.content).strip()
        if text:
            lines.append(f"{role}: {text[:_TAIL_CHARS]}")
    return lines
