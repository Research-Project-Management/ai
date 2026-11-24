"""Shared synthesis step and observation/peer-finding formatting helpers for tool agents."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ai.shared.contracts.run_control import RunUsage
from ai.shared.llm.factory import ChatRunnable
from ai.shared.llm.output import extract_text, strip_reasoning


def format_conversation_tail(tail: Sequence[str] | None) -> str:
    """Format recent conversation lines as an optional prompt prefix."""
    if not tail:
        return ""
    return "Recent conversation:\n" + "\n".join(tail) + "\n\n"


def format_peer_findings(peer_findings: Sequence[Mapping[str, Any]]) -> str:
    """Format prior findings from peer specialist agents, or empty string if none."""
    if not peer_findings:
        return ""
    lines: list[str] = ["## Findings from Peer Specialists"]
    for finding in peer_findings:
        agent = str(finding.get("agent") or "peer")
        content = str(finding.get("content") or "").strip()
        lines.append(f"### From `{agent}`\n{content}")
    return "\n\n".join(lines)


def format_tool_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    max_chars: int = 14_000,
) -> str:
    """Serialize compact tool observations with tail truncation."""
    if not observations:
        return "Tool observations: (none yet)"
    rendered = json.dumps(list(observations), ensure_ascii=False, indent=2)
    if len(rendered) > max_chars:
        rendered = rendered[-max_chars:]
    return f"Tool observations so far:\n{rendered}"


async def run_synthesize_step(
    llm: ChatRunnable,
    *,
    draft_answer: str | None,
    allow_draft_fast_path: bool = True,
    system_prompt: str,
    user_content: str,
    conversation_tail: Sequence[str] | None = None,
    prior_usage: RunUsage | None = None,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Run final response synthesis with optional draft fast-path and RunUsage merge."""
    draft = str(draft_answer or "").strip()
    if draft and allow_draft_fast_path:
        return {"final_answer": draft}

    full_user_content = format_conversation_tail(conversation_tail) + user_content
    response = await llm.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=full_user_content),
        ],
        config=config,
    )
    base_usage = prior_usage or RunUsage()
    return {
        "final_answer": strip_reasoning(extract_text(response.content)),
        "usage": base_usage.merge(RunUsage.from_message(response)),
    }
