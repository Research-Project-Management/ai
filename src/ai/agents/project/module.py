"""Assembled `project_agent` vertical-slice module."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from ai.agents.project.gateway.contracts import ProjectGateway
from ai.agents.project.gateway.http import HttpProjectGateway
from ai.agents.project.tools.contracts import (
    ProjectToolRegistry,
    build_default_project_tool_registry,
)
from ai.agents.project.workflow.graph import build_project_agent_spec
from ai.orchestrator.contracts import AgentSpec
from ai.shared.config.agents import ProjectConfig
from ai.shared.llm.factory import ChatRunnable
from ai.shared.workflow.plan import ToolAgentConfig


@dataclass(frozen=True)
class ProjectModule:
    """Assembled Flux Project Management dependencies exposed for Composition Root wiring."""

    gateway: ProjectGateway
    tool_registry: ProjectToolRegistry
    agent_spec: AgentSpec


def build_project_module(
    *,
    llm: ChatRunnable,
    config: ProjectConfig | None = None,
    agent_config: ToolAgentConfig | None = None,
    gateway: ProjectGateway | None = None,
    tool_registry: ProjectToolRegistry | None = None,
    http_client: httpx.AsyncClient | None = None,
    internal_api_key: str = "",
) -> ProjectModule:
    """Assemble Project gateway facade, tool registry, and `project_agent` AgentSpec."""
    cfg = config or ProjectConfig()
    active_gateway: ProjectGateway = gateway or HttpProjectGateway(
        base_url=cfg.backend_url,
        timeout_seconds=cfg.timeout_seconds,
        http_client=http_client,
        internal_api_key=internal_api_key,
    )
    active_tools = tool_registry or build_default_project_tool_registry()
    eff_agent_cfg = agent_config or ToolAgentConfig(
        max_steps=cfg.max_steps,
        tool_timeout_seconds=cfg.timeout_seconds,
        timeout_seconds=min(300.0, max(45.0, cfg.timeout_seconds * cfg.max_steps)),
        hitl_policy=cfg.hitl_policy,
    )
    spec = build_project_agent_spec(
        llm,
        active_gateway,
        tool_registry=active_tools,
        config=eff_agent_cfg,
    )
    return ProjectModule(
        gateway=active_gateway,
        tool_registry=active_tools,
        agent_spec=spec,
    )


__all__ = ["ProjectModule", "build_project_module"]
