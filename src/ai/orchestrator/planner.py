"""LLM-backed Supervisor planner and routing decision parser."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ai.orchestrator.context_policy import SlidingWindowContextPolicy
from ai.orchestrator.contracts import (
    AgentRegistry,
    AgentTaskAssignment,
    ContextWindowPolicy,
    RoutingDecision,
)
from ai.orchestrator.prompts import build_supervisor_prompt
from ai.orchestrator.state import RootState
from ai.shared.contracts.run_control import RunUsage
from ai.shared.llm.factory import ChatRunnable
from ai.shared.llm.output import extract_text, parse_json_object, with_nostream


def latest_user_query(messages: Sequence[BaseMessage]) -> str:
    """Return the trimmed text of the most recent HumanMessage, or empty string."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return extract_text(msg.content).strip()
    return ""


def parse_routing_decision(
    raw_text: str,
    registry: AgentRegistry,
    *,
    has_prior_outputs: bool = False,
) -> RoutingDecision:
    """Parse and validate Supervisor JSON output against registered specialist agents."""
    fallback_mode = "synthesize" if has_prior_outputs else "direct"
    payload = parse_json_object(raw_text)
    if payload is None:
        return RoutingDecision(mode=fallback_mode, reasoning=f"fallback to {fallback_mode}")

    mode = payload.get("mode")
    reasoning = str(payload.get("reasoning") or "")
    plan_summary = str(payload.get("plan_summary") or "").strip()
    needs_followup = bool(payload.get("needs_followup", False))
    raw_targets = payload.get("targets")

    if mode == "synthesize":
        return RoutingDecision(
            mode="synthesize" if has_prior_outputs else "direct",
            reasoning=reasoning or "synthesize",
        )

    if mode != "delegate" or not isinstance(raw_targets, list):
        return RoutingDecision(mode=fallback_mode, reasoning=reasoning or fallback_mode)

    seen: set[str] = set()
    valid_targets: list[AgentTaskAssignment] = []
    for item in raw_targets:
        if not isinstance(item, dict):
            continue
        agent_name = str(item.get("agent") or "").strip()
        instruction = str(item.get("instruction") or "").strip()
        if agent_name in registry.names and instruction and agent_name not in seen:
            seen.add(agent_name)
            valid_targets.append(AgentTaskAssignment(agent=agent_name, instruction=instruction))

    if not valid_targets:
        return RoutingDecision(mode=fallback_mode, reasoning=f"fallback to {fallback_mode}")

    return RoutingDecision(
        mode="delegate",
        targets=tuple(valid_targets),
        needs_followup=needs_followup,
        plan_summary=plan_summary,
        reasoning=reasoning,
    )


class LLMSupervisorPlanner:
    """Default RoutePlanner using the configured router ChatRunnable."""

    def __init__(
        self,
        router_llm: ChatRunnable,
        *,
        context_policy: ContextWindowPolicy | None = None,
        max_rounds: int = 3,
    ) -> None:
        self._router_llm = router_llm
        self._context_policy: ContextWindowPolicy = context_policy or SlidingWindowContextPolicy(
            max_messages=6, max_chars=12_000
        )
        self._max_rounds = max_rounds

    async def plan(
        self,
        state: RootState,
        registry: AgentRegistry,
        config: RunnableConfig,
    ) -> tuple[RoutingDecision, RunUsage]:
        messages = state.get("messages", [])
        prior_outputs = state.get("agent_outputs", [])
        latest_human = latest_user_query(messages)

        if not latest_human:
            return (
                RoutingDecision(
                    mode="synthesize" if prior_outputs else "direct",
                    reasoning="empty query",
                ),
                RunUsage(),
            )

        recent = self._context_policy.select_messages(messages)
        req_ctx = state.get("context")
        round_count = state.get("round_count", 1)
        system_prompt = build_supervisor_prompt(
            registry.format_catalog(),
            round_count=round_count,
            max_rounds=self._max_rounds,
            plan_summary=state.get("plan_summary", ""),
            prior_outputs=prior_outputs,
            request_context=req_ctx,
        )

        prompt_messages: list[BaseMessage] = [SystemMessage(content=system_prompt), *recent]
        response = await self._router_llm.ainvoke(prompt_messages, config=with_nostream(config))
        router_usage = RunUsage.from_message(response)
        raw_text = extract_text(response.content)
        decision = parse_routing_decision(
            raw_text,
            registry,
            has_prior_outputs=bool(prior_outputs),
        )
        return decision, router_usage
