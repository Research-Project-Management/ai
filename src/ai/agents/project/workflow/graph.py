"""Subgraph compiler and Root Orchestrator AgentSpec adapter for `project_agent`."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from ai.agents.project.gateway.contracts import ProjectGateway
from ai.agents.project.tools.contracts import ProjectToolRegistry
from ai.agents.project.tools.registry import build_default_project_tool_registry
from ai.agents.project.workflow.contracts import ProjectAgentConfig
from ai.agents.project.workflow.nodes import build_project_nodes
from ai.agents.project.workflow.state import ProjectState
from ai.orchestrator.contracts import AgentOutput, AgentSpec
from ai.orchestrator.state import RootState
from ai.shared.contracts.run_control import RunUsage
from ai.shared.llm.factory import ChatRunnable
from ai.shared.workflow.graph import compile_tool_agent_graph
from ai.shared.workflow.turn import extract_peer_findings, latest_query, recent_tail


def build_project_subgraph(
    llm: ChatRunnable,
    gateway: ProjectGateway,
    *,
    tool_registry: ProjectToolRegistry | None = None,
    config: ProjectAgentConfig | None = None,
) -> CompiledStateGraph:
    """Compile `hydrate_context -> plan_and_act -> policy_gate -> execute_and_verify -> synthesize_response`."""
    cfg = config or ProjectAgentConfig()
    registry = tool_registry or build_default_project_tool_registry()
    nodes = build_project_nodes(
        llm=llm,
        gateway=gateway,
        tool_registry=registry,
        agent_config=cfg,
    )

    return compile_tool_agent_graph(ProjectState, nodes, max_steps=cfg.max_steps)


def build_project_agent_spec(
    llm: ChatRunnable,
    gateway: ProjectGateway,
    *,
    tool_registry: ProjectToolRegistry | None = None,
    config: ProjectAgentConfig | None = None,
) -> AgentSpec:
    """Build the pluggable `project_agent` AgentSpec with entity resolution & HITL gates."""
    cfg = config or ProjectAgentConfig()
    subgraph = build_project_subgraph(
        llm,
        gateway,
        tool_registry=tool_registry,
        config=cfg,
    )

    async def _run_project_agent(state: RootState, run_config: RunnableConfig) -> AgentOutput:
        configurable: dict[str, Any] = dict(run_config.get("configurable") or {})
        context: dict[str, Any] = dict(state.get("context") or {})

        query = latest_query(state)
        if not query:
            return AgentOutput(
                agent="project_agent",
                content="Yêu cầu quản lý dự án đang trống.",
                status="error",
                error="empty query",
            )

        space_id = str(configurable.get("space_id") or context.get("space_id") or "").strip()
        raw_pid = context.get("project_id") or configurable.get("project_id")
        project_id = str(raw_pid).strip() if isinstance(raw_pid, str) and raw_pid.strip() else None
        user_id = str(context.get("user_id") or configurable.get("user_id") or "").strip()

        sub_input: ProjectState = {
            "query": query,
            "conversation_tail": recent_tail(list(state.get("messages") or ())),
            "peer_findings": extract_peer_findings(state, exclude_agent="project_agent"),
            "space_id": space_id,
            "project_id": project_id,
            "user_id": user_id,
            "raw_context": context,
            "step": 0,
            "pending_calls": [],
            "approved_calls": [],
            "observations": [],
            "widgets": [],
            "citations": [],
            "draft_answer": "",
            "done": False,
        }
        result = await subgraph.ainvoke(sub_input, config=run_config)
        observations = result.get("observations") or ()
        widgets = list(result.get("widgets") or ())
        raw_usage = result.get("usage")
        artifacts: dict[str, Any] = {
            "steps": int(result.get("step", 1)),
            "tool_calls_executed": len(observations),
        }
        if widgets:
            artifacts["widgets"] = widgets

        return AgentOutput(
            agent="project_agent",
            content=str(result.get("final_answer") or ""),
            status="ok",
            citations=tuple(result.get("citations") or ()),
            artifacts=artifacts,
            usage=raw_usage if isinstance(raw_usage, RunUsage) else RunUsage(),
            is_final=True,
        )

    return AgentSpec(
        name="project_agent",
        description=(
            "Manages Flux research projects, Kanban work items, assignments, labels, comments, "
            "team members, workload KPIs, overdue tracking, project health audits, and activity feeds."
        ),
        handler=_run_project_agent,
        capabilities=(
            "project_overview",
            "work_item_crud",
            "team_management",
            "workload_analytics",
            "health_audit",
            "hitl_actions",
        ),
        timeout_seconds=cfg.timeout_seconds,
    )
