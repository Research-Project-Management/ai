"""Shared planner contract: `PlannedCall`, `ActionPlan`, parsing, and one reasoning step."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field

from ai.shared.contracts.run_control import RunUsage
from ai.shared.llm.factory import ChatRunnable
from ai.shared.llm.nostream import with_nostream
from ai.shared.llm.output import extract_text, parse_json_object


class PlannedCall(BaseModel):
    """Single tool invocation emitted by the action planner."""

    model_config = ConfigDict(frozen=True)

    tool: str = Field(..., min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class ActionPlan(BaseModel):
    """Structured plan emitted on each reasoning step."""

    model_config = ConfigDict(frozen=True)

    action: Literal["call_tools", "respond"]
    tool_calls: tuple[PlannedCall, ...] = ()
    answer: str = ""
    reasoning: str = ""


def parse_action_plan(raw_text: str, *, max_calls: int) -> ActionPlan:
    """Parse planner JSON; any malformed output degrades to `respond` (synthesize)."""
    payload = parse_json_object(raw_text)
    if payload is None:
        return ActionPlan(action="respond", answer="", reasoning="fallback to synthesize")

    answer = str(payload.get("answer") or "").strip()
    reasoning = str(payload.get("reasoning") or "").strip()
    if payload.get("action") != "call_tools":
        return ActionPlan(action="respond", answer=answer, reasoning=reasoning)

    calls: list[PlannedCall] = []
    raw_calls = payload.get("tool_calls")
    if isinstance(raw_calls, list):
        for item in raw_calls[:max_calls]:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool") or "").strip()
            args = item.get("args")
            if tool:
                calls.append(
                    PlannedCall(
                        tool=tool,
                        args=dict(args) if isinstance(args, dict) else {},
                        reason=str(item.get("reason") or "").strip(),
                    )
                )

    if not calls:
        return ActionPlan(action="respond", answer=answer, reasoning=reasoning or "no tool calls")
    return ActionPlan(action="call_tools", tool_calls=tuple(calls), reasoning=reasoning)


async def run_plan_step(
    llm: ChatRunnable,
    *,
    system_prompt: str,
    user_content: str,
    config: RunnableConfig,
    step: int,
    prior_usage: RunUsage | None,
    max_calls: int,
) -> dict[str, Any]:
    """Run one planner LLM call and return the state delta shared by every tool-using agent."""
    response = await llm.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_content)],
        config=with_nostream(config),
    )
    plan = parse_action_plan(extract_text(response.content), max_calls=max_calls)
    usage = (prior_usage or RunUsage()).merge(RunUsage.from_message(response))

    if plan.action == "respond" or not plan.tool_calls:
        return {
            "step": step,
            "pending_calls": [],
            "approved_calls": [],
            "draft_answer": plan.answer,
            "done": True,
            "usage": usage,
        }
    return {
        "step": step,
        "pending_calls": [c.model_dump() for c in plan.tool_calls],
        "approved_calls": [],
        "draft_answer": "",
        "done": False,
        "usage": usage,
    }
