"""Shared planner contract: `PlannedCall`, `ActionPlan`, parsing, and one reasoning step."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field

from ai.shared.contracts.run_control import RunUsage
from ai.shared.llm.factory import ChatRunnable
from ai.shared.llm.nostream import with_nostream
from ai.shared.llm.output import extract_text, parse_json_object, strip_reasoning


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


def _extract_native_tool_calls(response: Any, *, max_calls: int) -> tuple[PlannedCall, ...]:
    """Extract LangChain native `AIMessage.tool_calls` into `PlannedCall` items."""
    raw_calls = getattr(response, "tool_calls", None)
    if not isinstance(raw_calls, list) or not raw_calls:
        return ()
    calls: list[PlannedCall] = []
    for item in raw_calls[:max_calls]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        args = item.get("args")
        if name:
            calls.append(
                PlannedCall(
                    tool=name,
                    args=dict(args) if isinstance(args, dict) else {},
                )
            )
    return tuple(calls)


async def run_plan_step(
    llm: ChatRunnable,
    *,
    system_prompt: str,
    user_content: str,
    config: RunnableConfig,
    step: int,
    prior_usage: RunUsage | None,
    max_calls: int,
    tools: Sequence[dict[str, Any]] | None = None,
    native_tools_bound: bool = False,
) -> dict[str, Any]:
    """Run one planner LLM call using native `bind_tools` when available, with JSON fallback."""
    bind_fn = getattr(llm, "bind_tools", None)
    used_native_tools = native_tools_bound
    runnable: Any = llm
    if tools and callable(bind_fn):
        try:
            runnable = bind_fn(tools)
            used_native_tools = True
        except (NotImplementedError, TypeError):
            runnable = llm
            used_native_tools = native_tools_bound

    response = await runnable.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_content)],
        config=with_nostream(config),
    )
    usage = (prior_usage or RunUsage()).merge(RunUsage.from_message(response))

    native_calls = _extract_native_tool_calls(response, max_calls=max_calls)
    if native_calls:
        return {
            "step": step,
            "pending_calls": [c.model_dump() for c in native_calls],
            "approved_calls": [],
            "draft_answer": "",
            "done": False,
            "usage": usage,
        }

    raw_text = extract_text(response.content)
    plan = parse_action_plan(raw_text, max_calls=max_calls)
    if plan.action == "call_tools" and plan.tool_calls:
        return {
            "step": step,
            "pending_calls": [c.model_dump() for c in plan.tool_calls],
            "approved_calls": [],
            "draft_answer": "",
            "done": False,
            "usage": usage,
        }

    draft_answer = plan.answer
    if not draft_answer and used_native_tools:
        # When native tool-calling finishes without invoking a tool, the model's
        # plain-text reply IS the final answer (cuts simple read turns from 3 -> 2 LLM calls).
        cleaned = strip_reasoning(raw_text).strip()
        if cleaned and parse_json_object(cleaned) is None:
            draft_answer = cleaned

    return {
        "step": step,
        "pending_calls": [],
        "approved_calls": [],
        "draft_answer": draft_answer,
        "done": True,
        "usage": usage,
    }
