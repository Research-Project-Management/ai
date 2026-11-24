"""Shared StateGraph for tool agents: `[hydrate] -> plan -> gate -> execute -> synthesize`."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ai.orchestrator.contracts import AgentOutput, AgentSpec
from ai.orchestrator.state import RootState
from ai.shared.contracts.run_control import RunUsage
from ai.shared.workflow.turn import latest_query, recent_tail

Node = Callable[..., Awaitable[dict[str, Any]]]
ExtraStateBuilder = Callable[[RootState, dict[str, Any], dict[str, Any]], dict[str, Any]]
ExtraArtifactsBuilder = Callable[[Mapping[str, Any]], dict[str, Any]]


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


def build_tool_agent_spec(
    *,
    name: str,
    description: str,
    capabilities: tuple[str, ...],
    timeout_seconds: float,
    subgraph: CompiledStateGraph,
    empty_query_message: str,
    build_extra_state: ExtraStateBuilder | None = None,
    build_extra_artifacts: ExtraArtifactsBuilder | None = None,
) -> AgentSpec:
    """Wrap a compiled tool-agent subgraph into a Root Orchestrator `AgentSpec`."""

    async def _run_agent(state: RootState, run_config: RunnableConfig) -> AgentOutput:
        configurable: dict[str, Any] = dict(run_config.get("configurable") or {})
        context: dict[str, Any] = dict(state.get("context") or {})

        query = latest_query(state)
        if not query:
            return AgentOutput(
                agent=name,
                content=empty_query_message,
                status="error",
                error="empty query",
            )

        space_id = str(configurable.get("space_id") or context.get("space_id") or "").strip()
        user_id = str(context.get("user_id") or configurable.get("user_id") or "").strip()
        sub_input: dict[str, Any] = {
            "query": query,
            "conversation_tail": recent_tail(list(state.get("messages") or ())),
            "space_id": space_id,
            "user_id": user_id,
            "step": 0,
            "pending_calls": [],
            "approved_calls": [],
            "observations": [],
            "citations": [],
            "draft_answer": "",
            "done": False,
        }
        if build_extra_state is not None:
            sub_input.update(build_extra_state(state, context, configurable))

        result: Mapping[str, Any] = await subgraph.ainvoke(sub_input, config=run_config)
        observations = result.get("observations") or ()
        raw_usage = result.get("usage")
        artifacts: dict[str, Any] = {
            "steps": int(result.get("step", 1)),
            "tool_calls_executed": len(observations),
        }
        if build_extra_artifacts is not None:
            artifacts.update(build_extra_artifacts(result))

        return AgentOutput(
            agent=name,
            content=str(result.get("final_answer") or ""),
            status="ok",
            citations=tuple(result.get("citations") or ()),
            artifacts=artifacts,
            usage=raw_usage if isinstance(raw_usage, RunUsage) else RunUsage(),
            is_final=True,
        )

    return AgentSpec(
        name=name,
        description=description,
        handler=_run_agent,
        capabilities=capabilities,
        timeout_seconds=timeout_seconds,
    )
