"""Shared StateGraph for tool agents: `[hydrate] -> plan -> gate -> execute -> synthesize`."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

Node = Callable[..., Awaitable[dict[str, Any]]]


def compile_tool_agent_graph(
    state_schema: type,
    nodes: Mapping[str, Node],
    *,
    max_steps: int,
) -> CompiledStateGraph:
    """Compile the plan/gate/execute loop shared by editor, library, and project agents.

    Required nodes: `plan_and_act`, `policy_gate`, `execute_and_verify`, `synthesize_response`.
    An optional `hydrate_context` node runs first when provided.
    """

    def route_after_plan(state: Mapping[str, Any]) -> str:
        if state.get("done", False) or not state.get("pending_calls"):
            return "synthesize"
        return "gate"

    def route_after_gate(state: Mapping[str, Any]) -> str:
        return "execute" if state.get("approved_calls") else "synthesize"

    def route_after_execute(state: Mapping[str, Any]) -> str:
        if state.get("done", False) or int(state.get("step", 0)) >= max_steps:
            return "synthesize"
        return "plan"

    builder = StateGraph(state_schema)
    for name, node in nodes.items():
        builder.add_node(name, node)

    entry = "hydrate_context" if "hydrate_context" in nodes else "plan_and_act"
    builder.add_edge(START, entry)
    if entry == "hydrate_context":
        builder.add_edge("hydrate_context", "plan_and_act")
    builder.add_conditional_edges(
        "plan_and_act",
        route_after_plan,
        {"gate": "policy_gate", "synthesize": "synthesize_response"},
    )
    builder.add_conditional_edges(
        "policy_gate",
        route_after_gate,
        {"execute": "execute_and_verify", "synthesize": "synthesize_response"},
    )
    builder.add_conditional_edges(
        "execute_and_verify",
        route_after_execute,
        {"plan": "plan_and_act", "synthesize": "synthesize_response"},
    )
    builder.add_edge("synthesize_response", END)
    return builder.compile()
