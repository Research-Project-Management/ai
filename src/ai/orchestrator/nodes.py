"""Node container encapsulating Supervisor, Synthesizer, Direct, and Guard nodes."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langgraph.types import Send

from ai.orchestrator.citations import select_indexed_citations
from ai.orchestrator.contracts import (
    AgentRegistry,
    AgentTaskAssignment,
    ContextWindowPolicy,
    DraftOutput,
    OutputGuard,
    RoutePlanner,
    RoutingDecision,
)
from ai.orchestrator.policies import DEFAULT_POLICIES, PolicyChain, PolicyContext
from ai.orchestrator.prompts import build_direct_system_prompt, build_synthesizer_prompt
from ai.orchestrator.state import RESET_AGENT_OUTPUTS, RESET_RUN_USAGE, RootState
from ai.shared.contracts.run_control import RunUsage
from ai.shared.llm.factory import ChatRunnable
from ai.shared.llm.nostream import with_nostream
from ai.shared.llm.output import extract_text

logger = logging.getLogger(__name__)


@runtime_checkable
class _UsageAwareRoutePlanner(Protocol):
    """Optional extension for RoutePlanners that also report router LLM token usage."""

    async def plan_with_usage(
        self,
        state: Any,
        registry: AgentRegistry,
        config: RunnableConfig,
    ) -> tuple[RoutingDecision, RunUsage]: ...


def _latest_user_query(messages: Sequence[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return extract_text(msg.content).strip()
    return ""


class RootGraphNodes:
    """Testable DI node suite wired by `build_root_graph`."""

    def __init__(
        self,
        *,
        llm: ChatRunnable,
        registry: AgentRegistry,
        planner: RoutePlanner,
        context_policy: ContextWindowPolicy,
        output_guards: Sequence[OutputGuard],
        has_blocking_guards: bool,
        max_rounds: int,
        policies: PolicyChain = DEFAULT_POLICIES,
        store: BaseStore | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._planner = planner
        self._context_policy = context_policy
        self._output_guards = tuple(output_guards)
        self._has_blocking_guards = has_blocking_guards
        self._max_rounds = max_rounds
        self._policies = policies
        self._store = store

    async def turn_init(self, state: RootState, config: RunnableConfig) -> dict[str, Any]:
        """Reset transient turn accumulators and optionally enrich workspace preferences."""
        updates: dict[str, Any] = {
            "agent_outputs": RESET_AGENT_OUTPUTS,
            "citations": [],
            "artifacts": {},
            "usage": RESET_RUN_USAGE,
            "final_response": "",
            "round_count": 0,
            "plan_summary": "",
            "executed_rounds": [],
        }
        if self._store is not None:
            configurable = dict(config.get("configurable") or {})
            raw_ctx = dict(state.get("context") or {})
            space_id = str(configurable.get("space_id") or raw_ctx.get("space_id") or "").strip()
            if space_id and "workspace_memory" not in raw_ctx:
                try:
                    mem_items = await self._store.asearch(("spaces", space_id, "prefs"), limit=5)
                    if mem_items:
                        raw_ctx["workspace_memory"] = {
                            item.key: dict(item.value) for item in mem_items
                        }
                        updates["context"] = raw_ctx
                except Exception as exc:  # noqa: BLE001 - fail-open memory enrichment
                    logger.debug("Long-term store preference lookup skipped: %s", exc)
        return updates

    async def supervisor(self, state: RootState, config: RunnableConfig) -> dict[str, Any]:
        """Plan round routing via intent hint or Supervisor `RoutePlanner`."""
        current_round = state.get("round_count", 0) + 1
        hint = (state.get("intent_hint") or "").strip()
        latest_query = _latest_user_query(state.get("messages", []))
        router_usage = RunUsage()

        if current_round == 1 and hint == "direct":
            decision = RoutingDecision(
                mode="direct",
                reasoning="explicit direct intent_hint",
            )
        elif current_round == 1 and hint and hint in self._registry.names:
            instruction = latest_query or hint
            decision = RoutingDecision(
                mode="delegate",
                targets=(AgentTaskAssignment(agent=hint, instruction=instruction),),
                needs_followup=False,
                reasoning=f"explicit intent_hint={hint}",
            )
        elif current_round == 1 and self._registry.is_empty:
            decision = RoutingDecision(
                mode="direct",
                reasoning="no specialist agents registered",
            )
        else:
            planner_state = {**state, "round_count": current_round}
            if isinstance(self._planner, _UsageAwareRoutePlanner):
                decision, router_usage = await self._planner.plan_with_usage(
                    planner_state,
                    self._registry,
                    config,
                )
            else:
                decision = await self._planner.plan(
                    planner_state,
                    self._registry,
                    config,
                )

        decision = self._policies.apply(
            decision,
            PolicyContext(
                query=latest_query,
                request_context=state.get("context"),
                prior_outputs=state.get("agent_outputs", []),
                registry=self._registry,
            ),
        )

        prev_rounds = list(state.get("executed_rounds") or [])
        if decision.mode == "delegate" and decision.targets:
            prev_rounds.append(",".join(t.agent for t in decision.targets))
        route = "->".join(prev_rounds) if prev_rounds else "direct"
        plan_summary = decision.plan_summary or state.get("plan_summary", "")

        updates: dict[str, Any] = {
            "round_count": current_round,
            "routing_decision": decision,
            "plan_summary": plan_summary,
            "executed_rounds": prev_rounds,
            "route": route,
        }
        if (
            router_usage.total_tokens > 0
            or router_usage.input_tokens > 0
            or router_usage.output_tokens > 0
        ):
            updates["usage"] = router_usage
        return updates

    def dispatch_from_supervisor(self, state: RootState) -> str | list[Send]:
        """Fan out Send commands to selected specialists, or route to direct/synthesizer."""
        decision = state.get("routing_decision")
        if decision is None or decision.mode != "delegate" or not decision.targets:
            return "synthesizer" if state.get("agent_outputs") else "direct_respond"
        return [
            Send(
                target.agent,
                {
                    **state,
                    "active_agent": target.agent,
                    "active_instruction": target.instruction,
                },
            )
            for target in decision.targets
        ]

    async def round_gate(self, state: RootState) -> dict[str, Any]:
        """Synchronization barrier after parallel specialist execution."""
        del state
        return {}

    def route_after_round(self, state: RootState) -> str:
        """Continue to next supervisor round when follow-up is needed within `max_rounds`."""
        decision = state.get("routing_decision")
        current_round = state.get("round_count", 1)
        if decision is not None and decision.needs_followup and current_round < self._max_rounds:
            return "supervisor"
        return "synthesizer"

    async def direct_respond(self, state: RootState, config: RunnableConfig) -> dict[str, Any]:
        """Generate a direct chat answer without specialist delegation."""
        windowed = self._context_policy.select_messages(state.get("messages", []))
        direct_messages: list[BaseMessage] = [
            SystemMessage(content=build_direct_system_prompt(state.get("context"))),
            *windowed,
        ]
        call_cfg = with_nostream(config) if self._has_blocking_guards else config
        response = await self._llm.ainvoke(direct_messages, config=call_cfg)
        return {
            "final_response": extract_text(response.content),
            "route": "direct",
            "usage": RunUsage.from_message(response),
        }

    async def synthesizer(self, state: RootState, config: RunnableConfig) -> dict[str, Any]:
        """Aggregate specialist outputs and synthesize the final cited response."""
        outputs = state.get("agent_outputs", [])
        merged_citations = [cite for out in outputs for cite in out.citations]
        merged_artifacts: dict[str, Any] = {
            key: value for out in outputs for key, value in out.artifacts.items()
        }

        turn_usage = RunUsage()
        if len(outputs) == 1 and outputs[0].is_final and outputs[0].status == "ok":
            content = outputs[0].content
        else:
            windowed = self._context_policy.select_messages(state.get("messages", []))
            synth_messages: list[BaseMessage] = [
                SystemMessage(content=build_synthesizer_prompt(outputs)),
                *windowed,
            ]
            call_cfg = with_nostream(config) if self._has_blocking_guards else config
            response = await self._llm.ainvoke(synth_messages, config=call_cfg)
            content = extract_text(response.content)
            turn_usage = RunUsage.from_message(response)
            merged_citations = select_indexed_citations(content, merged_citations)

        return {
            "final_response": content,
            "citations": merged_citations,
            "artifacts": merged_artifacts,
            "usage": turn_usage,
        }

    async def output_guard(self, state: RootState, config: RunnableConfig) -> dict[str, Any]:
        """Run sequential `OutputGuard` validators over draft text, citations, and artifacts."""
        draft = DraftOutput(
            content=state.get("final_response", ""),
            citations=tuple(state.get("citations") or ()),
            artifacts=dict(state.get("artifacts") or {}),
            agent_outputs=tuple(state.get("agent_outputs") or ()),
        )
        for guard in self._output_guards:
            draft = await guard.validate(draft, state, config)

        return {
            "messages": [AIMessage(content=draft.content)],
            "final_response": draft.content,
            "citations": list(draft.citations),
            "artifacts": dict(draft.artifacts),
        }


__all__ = ["RootGraphNodes"]
