"""Root StateGraph assembly for the multi-agent orchestrator."""

import asyncio
import logging
from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from ai.orchestrator.context_policy import SlidingWindowContextPolicy
from ai.orchestrator.contracts import (
    AgentOutput,
    AgentRegistry,
    AgentSpec,
    CitationItem,
    ContextWindowPolicy,
    OutputGuard,
    RoutePlanner,
)
from ai.orchestrator.nodes import RootGraphNodes
from ai.orchestrator.planner import LLMSupervisorPlanner
from ai.orchestrator.state import RootState
from ai.shared.contracts.run_control import RunUsage
from ai.shared.llm.factory import ChatRunnable
from ai.shared.llm.output import extract_text, with_nostream

logger = logging.getLogger(__name__)


class SpecialistNode(Protocol):
    async def __call__(self, state: RootState, config: RunnableConfig) -> dict[str, Any]: ...


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
        not bool(getattr(guard, "supports_live_streaming", False)) for guard in active_guards
    )
    active_planner: RoutePlanner = planner or LLMSupervisorPlanner(
        router_llm or llm,
        context_policy=active_context_policy,
        max_rounds=max_rounds,
    )

    nodes = RootGraphNodes(
        llm=llm,
        registry=active_registry,
        planner=active_planner,
        context_policy=active_context_policy,
        output_guards=active_guards,
        has_blocking_guards=has_blocking_guards,
        max_rounds=max_rounds,
        store=store,
    )

    builder = StateGraph(RootState)
    builder.add_node("turn_init", nodes.turn_init)
    builder.add_node("supervisor", nodes.supervisor)
    builder.add_node("direct_respond", nodes.direct_respond)
    builder.add_node("synthesizer", nodes.synthesizer)
    builder.add_node("output_guard", nodes.output_guard)

    builder.add_edge(START, "turn_init")
    builder.add_edge("turn_init", "supervisor")
    builder.add_conditional_edges("supervisor", nodes.dispatch_from_supervisor)

    if not active_registry.is_empty:
        builder.add_node("round_gate", nodes.round_gate)
        for spec in active_registry.specs:
            builder.add_node(spec.name, _make_specialist_node(spec, has_guards=has_blocking_guards))
            builder.add_edge(spec.name, "round_gate")
        builder.add_conditional_edges(
            "round_gate",
            nodes.route_after_round,
            {"supervisor": "supervisor", "synthesizer": "synthesizer"},
        )

    builder.add_edge("direct_respond", "output_guard")
    builder.add_edge("synthesizer", "output_guard")
    builder.add_edge("output_guard", END)
    return builder.compile(checkpointer=checkpointer, store=store)
