"""LLM-backed Supervisor planner and internal streaming helpers."""


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
from ai.shared.context.contracts import extract_viewport_focus
from ai.shared.context.facade import ResearchContextFacade
from ai.shared.llm.factory import ChatRunnable
from ai.shared.llm.nostream import with_nostream
from ai.shared.llm.output import extract_text, parse_json_object
from ai.shared.resilience.cache import ToolCache, build_cache_key


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
    """Default RoutePlanner using the configured router ChatRunnable and a ContextWindowPolicy."""

    def __init__(
        self,
        router_llm: ChatRunnable,
        *,
        context_policy: ContextWindowPolicy | None = None,
        context_facade: ResearchContextFacade | None = None,
        cache: ToolCache | None = None,
        cache_ttl_seconds: int = 300,
        max_rounds: int = 3,
    ) -> None:
        self._router_llm = router_llm
        self._context_policy: ContextWindowPolicy = (
            context_policy or SlidingWindowContextPolicy(max_messages=6, max_chars=12_000)
        )
        self._context_facade = context_facade
        self._cache = cache
        self._cache_ttl = max(30, cache_ttl_seconds)
        self._max_rounds = max_rounds

    async def plan(
        self,
        state: RootState,
        registry: AgentRegistry,
        config: RunnableConfig,
    ) -> RoutingDecision:
        messages = state.get("messages", [])
        prior_outputs = state.get("agent_outputs", [])
        latest_human = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                latest_human = extract_text(msg.content).strip()
                break

        if not latest_human:
            return RoutingDecision(
                mode="synthesize" if prior_outputs else "direct",
                reasoning="empty query",
            )

        recent = self._context_policy.select_messages(messages)
        req_ctx = state.get("context")
        envelope = None
        if self._context_facade is not None and req_ctx:
            focus = extract_viewport_focus(req_ctx)
            title = str(req_ctx.get("project_title") or req_ctx.get("title") or "").strip()
            venue = str(req_ctx.get("target_venue") or req_ctx.get("venue") or "").strip() or None
            deadline = str(req_ctx.get("target_deadline") or req_ctx.get("deadline") or "").strip() or None
            hypothesis = str(req_ctx.get("primary_hypothesis") or req_ctx.get("hypothesis") or "").strip() or None

            envelope = self._context_facade.assemble_context(
                focus,
                latest_human,
                research_title=title or "Research Project",
                target_venue=venue,
                deadline=deadline,
                hypothesis=hypothesis,
            )

        round_count = state.get("round_count", 1)
        system_prompt = build_supervisor_prompt(
            registry.format_catalog(),
            round_count=round_count,
            max_rounds=self._max_rounds,
            plan_summary=state.get("plan_summary", ""),
            prior_outputs=prior_outputs,
            request_context=req_ctx,
            context_envelope=envelope,
        )

        cache_key = ""
        if self._cache is not None and round_count == 1 and not prior_outputs:
            serialized_recent = [
                f"{msg.type}:{extract_text(msg.content)}" for msg in recent
            ]
            cache_key = build_cache_key(
                "router_plan",
                {"sys": system_prompt, "msgs": serialized_recent},
            )
            cached = await self._cache.get(cache_key)
            if isinstance(cached, dict) and "raw_text" in cached:
                return parse_routing_decision(
                    str(cached["raw_text"]),
                    registry,
                    has_prior_outputs=False,
                )

        prompt_messages: list[BaseMessage] = [SystemMessage(content=system_prompt), *recent]
        response = await self._router_llm.ainvoke(prompt_messages, config=with_nostream(config))
        raw_text = extract_text(response.content)
        decision = parse_routing_decision(
            raw_text,
            registry,
            has_prior_outputs=bool(prior_outputs),
        )
        if self._cache is not None and cache_key and not decision.reasoning.startswith("fallback to"):
            await self._cache.set(
                cache_key,
                {"raw_text": raw_text},
                ttl_seconds=self._cache_ttl,
            )
        return decision
