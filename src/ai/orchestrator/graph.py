"""Root StateGraph assembly for the multi-agent orchestrator."""

import asyncio
import logging
from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Send

from ai.orchestrator.citations import select_indexed_citations
from ai.orchestrator.context_policy import SlidingWindowContextPolicy
from ai.orchestrator.contracts import (
    AgentOutput,
    AgentRegistry,
    AgentSpec,
    AgentTaskAssignment,
    CitationItem,
    ContextWindowPolicy,
    DraftOutput,
    OutputGuard,
    RoutePlanner,
    RoutingDecision,
)
from ai.orchestrator.planner import LLMSupervisorPlanner
from ai.orchestrator.prompts import build_synthesizer_prompt
from ai.orchestrator.state import RESET_AGENT_OUTPUTS, RESET_RUN_USAGE, RootState
from ai.shared.context.facade import ResearchContextFacade
from ai.shared.contracts.run_control import RunUsage
from ai.shared.llm.factory import ChatRunnable
from ai.shared.llm.nostream import with_nostream
from ai.shared.llm.output import extract_text
from ai.shared.resilience.cache import ToolCache

logger = logging.getLogger(__name__)


class SpecialistNode(Protocol):
    async def __call__(self, state: RootState, config: RunnableConfig) -> dict[str, Any]: ...


def _latest_user_query(messages: Sequence[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return extract_text(msg.content).strip()
    return ""


def _normalize_citations(raw_citations: Any) -> tuple[CitationItem, ...]:
    if not raw_citations:
        return ()
    items: list[CitationItem] = []
    for item in raw_citations:
        if isinstance(item, CitationItem):
            items.append(item)
        elif isinstance(item, dict):
            items.append(CitationItem.model_validate(item))
    return tuple(items)


def _normalize_agent_output(agent_name: str, raw: Any) -> AgentOutput:
    if isinstance(raw, AgentOutput):
        return raw
    if isinstance(raw, str):
        return AgentOutput(agent=agent_name, content=raw)
    if isinstance(raw, dict):
        content = raw.get("final_response") or raw.get("content")
        if content is None:
            msgs = raw.get("messages")
            if isinstance(msgs, list) and msgs:
                last_msg = msgs[-1]
                msg_content = (
                    last_msg.get("content", "")
                    if isinstance(last_msg, dict)
                    else getattr(last_msg, "content", "")
                )
                content = extract_text(msg_content)
            else:
                content = ""
        elif not isinstance(content, str):
            content = extract_text(content)

        status = "error" if raw.get("status") == "error" else "ok"
        error = str(raw["error"]) if raw.get("error") is not None else None
        raw_usage = raw.get("usage")
        return AgentOutput(
            agent=agent_name,
            content=content,
            status=status,
            error=error,
            citations=_normalize_citations(raw.get("citations")),
            artifacts=dict(raw.get("artifacts") or {}),
            usage=raw_usage if isinstance(raw_usage, RunUsage) else RunUsage(),
            is_final=bool(raw.get("is_final", False)),
        )
    return AgentOutput(agent=agent_name, content=str(raw))


def _make_specialist_node(spec: AgentSpec, *, has_guards: bool = False) -> SpecialistNode:
    async def _run_specialist(state: RootState, config: RunnableConfig) -> dict[str, Any]:
        decision = state.get("routing_decision")
        can_stream = (
            not has_guards
            and decision is not None
            and len(decision.targets) == 1
            and not decision.needs_followup
        )
        call_config = config if can_stream else with_nostream(config)
        handler = spec.handler
        try:
            coro = (
                handler.ainvoke(state, config=call_config)
                if hasattr(handler, "ainvoke")
                else handler(state, call_config)
            )
            raw_result = await asyncio.wait_for(coro, timeout=spec.timeout_seconds)
            out = _normalize_agent_output(spec.name, raw_result)
            return {"agent_outputs": [out], "usage": out.usage}
        except GraphBubbleUp:
            raise
        except TimeoutError:
            return {
                "agent_outputs": [
                    AgentOutput(
                        agent=spec.name,
                        content="",
                        status="error",
                        error=f"TimeoutError: specialist exceeded {spec.timeout_seconds}s",
                    )
                ]
            }
        except Exception as exc:
            logger.warning("Specialist '%s' failed: %s", spec.name, exc, exc_info=True)
            return {
                "agent_outputs": [
                    AgentOutput(
                        agent=spec.name,
                        content="",
                        status="error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                ]
            }

    return _run_specialist


def build_root_graph(
    *,
    llm: ChatRunnable,
    router_llm: ChatRunnable | None = None,
    registry: AgentRegistry | None = None,
    planner: RoutePlanner | None = None,
    context_policy: ContextWindowPolicy | None = None,
    context_facade: ResearchContextFacade | None = None,
    router_cache: ToolCache | None = None,
    output_guards: Sequence[OutputGuard] = (),
    max_rounds: int = 3,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
) -> CompiledStateGraph:
    """Build and compile the root orchestration StateGraph."""
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")

    active_registry = registry or AgentRegistry()
    active_context_policy: ContextWindowPolicy = context_policy or SlidingWindowContextPolicy()
    active_guards = tuple(output_guards)
    has_blocking_guards = any(
        not bool(getattr(guard, "supports_live_streaming", False))
        for guard in active_guards
    )
    active_planner: RoutePlanner = planner or LLMSupervisorPlanner(
        router_llm or llm,
        context_policy=active_context_policy,
        context_facade=context_facade,
        cache=router_cache,
        max_rounds=max_rounds,
    )

    async def turn_init(state: RootState, config: RunnableConfig) -> dict[str, Any]:
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
        if store is not None:
            configurable = dict(config.get("configurable") or {})
            raw_ctx = dict(state.get("context") or {})
            space_id = str(configurable.get("space_id") or raw_ctx.get("space_id") or "").strip()
            if space_id and "workspace_memory" not in raw_ctx:
                try:
                    mem_items = await store.asearch(("spaces", space_id, "prefs"), limit=5)
                    if mem_items:
                        raw_ctx["workspace_memory"] = {
                            item.key: dict(item.value) for item in mem_items
                        }
                        updates["context"] = raw_ctx
                except Exception as exc:  # noqa: BLE001 - fail-open memory enrichment
                    logger.debug("Long-term store preference lookup skipped: %s", exc)
        return updates

    async def supervisor(state: RootState, config: RunnableConfig) -> dict[str, Any]:
        current_round = state.get("round_count", 0) + 1
        hint = (state.get("intent_hint") or "").strip()

        if current_round == 1:
            if hint == "direct":
                decision = RoutingDecision(
                    mode="direct",
                    reasoning="explicit direct intent_hint",
                )
            elif hint and hint in active_registry.names:
                instruction = _latest_user_query(state.get("messages", [])) or hint
                decision = RoutingDecision(
                    mode="delegate",
                    targets=(AgentTaskAssignment(agent=hint, instruction=instruction),),
                    needs_followup=False,
                    reasoning=f"explicit intent_hint={hint}",
                )
            elif active_registry.is_empty:
                decision = RoutingDecision(
                    mode="direct",
                    reasoning="no specialist agents registered",
                )
            else:
                decision = await active_planner.plan(
                    {**state, "round_count": current_round},
                    active_registry,
                    config,
                )
        else:
            decision = await active_planner.plan(
                {**state, "round_count": current_round},
                active_registry,
                config,
            )

        prev_rounds = list(state.get("executed_rounds") or [])
        if decision.mode == "delegate" and decision.targets:
            prev_rounds.append(",".join(t.agent for t in decision.targets))
        route = "->".join(prev_rounds) if prev_rounds else "direct"
        plan_summary = decision.plan_summary or state.get("plan_summary", "")

        return {
            "round_count": current_round,
            "routing_decision": decision,
            "plan_summary": plan_summary,
            "executed_rounds": prev_rounds,
            "route": route,
        }

    def _dispatch_from_supervisor(state: RootState) -> str | list[Send]:
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

    async def round_gate(state: RootState) -> dict[str, Any]:
        return {}

    def _route_after_round(state: RootState) -> str:
        decision = state.get("routing_decision")
        current_round = state.get("round_count", 1)
        if decision is not None and decision.needs_followup and current_round < max_rounds:
            return "supervisor"
        return "synthesizer"

    async def direct_respond(state: RootState, config: RunnableConfig) -> dict[str, Any]:
        windowed = active_context_policy.select_messages(state.get("messages", []))
        call_cfg = with_nostream(config) if has_blocking_guards else config
        response = await llm.ainvoke(windowed, config=call_cfg)
        return {
            "final_response": extract_text(response.content),
            "route": "direct",
            "usage": RunUsage.from_message(response),
        }

    async def synthesizer(state: RootState, config: RunnableConfig) -> dict[str, Any]:
        outputs = state.get("agent_outputs", [])
        merged_citations = [cite for out in outputs for cite in out.citations]
        merged_artifacts: dict[str, Any] = {
            key: value for out in outputs for key, value in out.artifacts.items()
        }

        turn_usage = RunUsage()
        if len(outputs) == 1 and outputs[0].is_final and outputs[0].status == "ok":
            content = outputs[0].content
        else:
            windowed = active_context_policy.select_messages(state.get("messages", []))
            synth_messages: list[BaseMessage] = [
                SystemMessage(content=build_synthesizer_prompt(outputs)),
                *windowed,
            ]
            call_cfg = with_nostream(config) if has_blocking_guards else config
            response = await llm.ainvoke(synth_messages, config=call_cfg)
            content = extract_text(response.content)
            turn_usage = RunUsage.from_message(response)
            merged_citations = select_indexed_citations(content, merged_citations)

        return {
            "final_response": content,
            "citations": merged_citations,
            "artifacts": merged_artifacts,
            "usage": turn_usage,
        }

    async def output_guard(state: RootState, config: RunnableConfig) -> dict[str, Any]:
        draft = DraftOutput(
            content=state.get("final_response", ""),
            citations=tuple(state.get("citations") or ()),
            artifacts=dict(state.get("artifacts") or {}),
            agent_outputs=tuple(state.get("agent_outputs") or ()),
        )
        for guard in active_guards:
            draft = await guard.validate(draft, state, config)

        return {
            "messages": [AIMessage(content=draft.content)],
            "final_response": draft.content,
            "citations": list(draft.citations),
            "artifacts": dict(draft.artifacts),
        }

    builder = StateGraph(RootState)
    builder.add_node("turn_init", turn_init)
    builder.add_node("supervisor", supervisor)
    builder.add_node("direct_respond", direct_respond)
    builder.add_node("synthesizer", synthesizer)
    builder.add_node("output_guard", output_guard)

    builder.add_edge(START, "turn_init")
    builder.add_edge("turn_init", "supervisor")
    builder.add_conditional_edges("supervisor", _dispatch_from_supervisor)

    if not active_registry.is_empty:
        builder.add_node("round_gate", round_gate)
        for spec in active_registry.specs:
            builder.add_node(spec.name, _make_specialist_node(spec, has_guards=has_blocking_guards))
            builder.add_edge(spec.name, "round_gate")
        builder.add_conditional_edges(
            "round_gate",
            _route_after_round,
            {"supervisor": "supervisor", "synthesizer": "synthesizer"},
        )

    builder.add_edge("direct_respond", "output_guard")
    builder.add_edge("synthesizer", "output_guard")
    builder.add_edge("output_guard", END)
    return builder.compile(checkpointer=checkpointer, store=store)
