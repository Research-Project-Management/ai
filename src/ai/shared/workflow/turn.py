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


def extract_peer_findings(
    state: Mapping[str, Any],
    *,
    exclude_agent: str,
    include_citations: bool = False,
    max_chars: int = 2500,
) -> list[dict[str, Any]]:
    """Collect successful peer agent outputs from `RootState` for cross-agent grounding."""
    outputs = state.get("agent_outputs") or ()
    findings: list[dict[str, Any]] = []
    for out in outputs:
        if out.agent == exclude_agent or out.status != "ok":
            continue
        snippet = out.content.strip()
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "\n...[truncated]"
        cite_summaries = (
            [
                {
                    "source_id": c.source_id,
                    "title": c.title,
                    "url": c.url,
                    "page": c.page,
                }
                for c in out.citations[:10]
            ]
            if include_citations
            else []
        )
        if snippet or cite_summaries:
            entry: dict[str, Any] = {"agent": out.agent, "content": snippet}
            if cite_summaries:
                entry["citations"] = cite_summaries
            findings.append(entry)
    return findings


__all__ = ["extract_peer_findings", "latest_query", "recent_tail"]
