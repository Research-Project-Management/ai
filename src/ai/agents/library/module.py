"""Assembled `library_agent` vertical-slice module."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from ai.agents.library.gateway.contracts import LibraryGateway
from ai.agents.library.gateway.http import HttpLibraryGateway
from ai.agents.library.tools.contracts import LibraryToolRegistry, build_default_tool_registry
from ai.agents.library.workflow.graph import build_library_agent_spec
from ai.orchestrator.contracts import AgentSpec
from ai.shared.config.agents import LibraryConfig
from ai.shared.llm.factory import ChatRunnable
from ai.shared.workflow.events import ToolEventPublisher
from ai.shared.workflow.plan import ToolAgentConfig


@dataclass(frozen=True)
class LibraryModule:
    """Assembled Flux Library dependencies exposed for Composition Root wiring."""

    gateway: LibraryGateway
    tool_registry: LibraryToolRegistry
    agent_spec: AgentSpec


def build_library_module(
    *,
    llm: ChatRunnable,
    config: LibraryConfig | None = None,
    agent_config: ToolAgentConfig | None = None,
    gateway: LibraryGateway | None = None,
    tool_registry: LibraryToolRegistry | None = None,
    http_client: httpx.AsyncClient | None = None,
    event_publisher: ToolEventPublisher | None = None,
) -> LibraryModule:
    """Assemble Library gateway adapter, tool registry, and `library_agent` AgentSpec."""
    cfg = config or LibraryConfig()
    active_gateway: LibraryGateway = gateway or HttpLibraryGateway(
        base_url=cfg.backend_url,
        timeout_seconds=cfg.timeout_seconds,
        max_items_per_call=cfg.max_items_per_call,
        http_client=http_client,
    )
    active_tools = tool_registry or build_default_tool_registry()
    eff_agent_cfg = agent_config or ToolAgentConfig(
        max_steps=cfg.max_steps,
        tool_timeout_seconds=cfg.timeout_seconds,
        timeout_seconds=min(300.0, max(30.0, cfg.timeout_seconds * cfg.max_steps)),
        hitl_policy=cfg.hitl_policy,
    )
    spec = build_library_agent_spec(
        llm,
        active_gateway,
        tool_registry=active_tools,
        config=eff_agent_cfg,
        event_publisher=event_publisher,
    )
    return LibraryModule(
        gateway=active_gateway,
        tool_registry=active_tools,
        agent_spec=spec,
    )


__all__ = ["LibraryModule", "build_library_module"]
