"""Subgraph compiler and Root Orchestrator AgentSpec adapter for `project_agent`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from ai.agents.project.gateway.contracts import ProjectGateway
from ai.agents.project.tools.contracts import (
    ProjectToolRegistry,
    build_default_project_tool_registry,
)
from ai.agents.project.workflow.nodes import build_project_nodes
from ai.agents.project.workflow.state import ProjectState
from ai.orchestrator.contracts import AgentSpec
from ai.orchestrator.state import RootState
from ai.shared.llm.factory import ChatRunnable
from ai.shared.workflow.graph import build_tool_agent_spec, compile_tool_agent_graph
from ai.shared.workflow.plan import ToolAgentConfig
from ai.shared.workflow.turn import extract_peer_findings


def build_project_subgraph(
    llm: ChatRunnable,
    gateway: ProjectGateway,
    *,
    tool_registry: ProjectToolRegistry | None = None,
    config: ToolAgentConfig | None = None,
) -> CompiledStateGraph:
    """Compile `hydrate_context -> plan_and_act -> policy_gate -> execute_and_verify -> synthesize_response`."""
    cfg = config or ToolAgentConfig()
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
    config: ToolAgentConfig | None = None,
) -> AgentSpec:
    """Build the pluggable `project_agent` AgentSpec with entity resolution & HITL gates."""
    cfg = config or ToolAgentConfig()
    subgraph = build_project_subgraph(
        llm,
        gateway,
        tool_registry=tool_registry,
        config=cfg,
    )

    def _extra_state(
        state: RootState, context: dict[str, Any], configurable: dict[str, Any]
    ) -> dict[str, Any]:
        raw_pid = context.get("project_id") or configurable.get("project_id")
        project_id = str(raw_pid).strip() if isinstance(raw_pid, str) and raw_pid.strip() else None
        return {
            "peer_findings": extract_peer_findings(state, exclude_agent="project_agent"),
            "project_id": project_id,
            "raw_context": context,
            "widgets": [],
        }

    def _extra_artifacts(result: Mapping[str, Any]) -> dict[str, Any]:
        widgets = list(result.get("widgets") or ())
        return {"widgets": widgets} if widgets else {}

    return build_tool_agent_spec(
        name="project_agent",
        description=(
            "Manages Flux research projects, Kanban work items, assignments, labels, comments, "
            "team members, workload KPIs, overdue tracking, project health audits, and activity feeds."
        ),
        capabilities=(
            "project_overview",
            "work_item_crud",
            "team_management",
            "workload_analytics",
            "health_audit",
            "hitl_actions",
        ),
        timeout_seconds=cfg.timeout_seconds,
        subgraph=subgraph,
        empty_query_message="Yêu cầu quản lý dự án đang trống.",
        build_extra_state=_extra_state,
        build_extra_artifacts=_extra_artifacts,
    )
