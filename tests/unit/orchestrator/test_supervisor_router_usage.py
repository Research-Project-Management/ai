"""Seam tests verifying Supervisor router RunUsage is accumulated into RootState.usage."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from ai.orchestrator.contracts import (
    AgentOutput,
    AgentRegistry,
    AgentSpec,
    RoutingDecision,
)
from ai.orchestrator.graph import build_root_graph
from ai.orchestrator.state import RootState
from ai.shared.contracts.run_control import RunUsage


async def test_root_graph_includes_router_usage_in_delegated_turn() -> None:
    def _router(_inp: LanguageModelInput) -> AIMessage:
        return AIMessage(
            content=json.dumps(
                {
                    "mode": "delegate",
                    "targets": [{"agent": "specialist", "instruction": "Inspect findings"}],
                    "reasoning": "needs specialist",
                }
            ),
            usage_metadata={"input_tokens": 50, "output_tokens": 20, "total_tokens": 70},
        )

    def _main(_inp: LanguageModelInput) -> AIMessage:
        return AIMessage(content="unused")

    async def _specialist(_state: RootState, _config: RunnableConfig) -> AgentOutput:
        return AgentOutput(
            agent="specialist",
            content="Specialist answer",
            usage=RunUsage(input_tokens=100, output_tokens=40, total_tokens=140),
            is_final=True,
        )

    registry = AgentRegistry(
        [AgentSpec(name="specialist", description="Test specialist", handler=_specialist)]
    )
    graph = build_root_graph(
        llm=RunnableLambda(_main),
        router_llm=RunnableLambda(_router),
        registry=registry,
    )

    result = await graph.ainvoke({"messages": [HumanMessage(content="Inspect my paper")]})
    usage = result["usage"]
    assert usage.input_tokens == 150
    assert usage.output_tokens == 60
    assert usage.total_tokens == 210


async def test_root_graph_includes_router_usage_in_direct_turn() -> None:
    def _router(_inp: LanguageModelInput) -> AIMessage:
        return AIMessage(
            content=json.dumps({"mode": "direct", "reasoning": "general greeting"}),
            usage_metadata={"input_tokens": 30, "output_tokens": 10, "total_tokens": 40},
        )

    def _main(_inp: LanguageModelInput) -> AIMessage:
        return AIMessage(
            content="Hello!",
            usage_metadata={"input_tokens": 80, "output_tokens": 25, "total_tokens": 105},
        )

    async def _noop(_state: RootState, _config: RunnableConfig) -> AgentOutput:
        return AgentOutput(agent="helper", content="unused", is_final=True)

    registry = AgentRegistry([AgentSpec(name="helper", description="Helper", handler=_noop)])
    graph = build_root_graph(
        llm=RunnableLambda(_main),
        router_llm=RunnableLambda(_router),
        registry=registry,
    )

    result = await graph.ainvoke({"messages": [HumanMessage(content="Hi there")]})
    usage = result["usage"]
    assert usage.input_tokens == 110
    assert usage.output_tokens == 35
    assert usage.total_tokens == 145


async def test_custom_route_planner_without_usage_method_remains_compatible() -> None:
    class _PlainPlanner:
        async def plan(
            self,
            state: Any,
            registry: AgentRegistry,
            config: RunnableConfig,
        ) -> RoutingDecision:
            del state, registry, config
            return RoutingDecision(mode="direct", reasoning="plain planner")

    def _main(_inp: LanguageModelInput) -> AIMessage:
        return AIMessage(
            content="Direct reply",
            usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
        )

    async def _noop(_state: RootState, _config: RunnableConfig) -> AgentOutput:
        return AgentOutput(agent="helper", content="unused", is_final=True)

    registry = AgentRegistry([AgentSpec(name="helper", description="Helper", handler=_noop)])
    graph = build_root_graph(
        llm=RunnableLambda(_main),
        planner=_PlainPlanner(),
        registry=registry,
    )

    result = await graph.ainvoke({"messages": [HumanMessage(content="Hello")]})
    assert result["final_response"] == "Direct reply"
    assert result["usage"].total_tokens == 20
