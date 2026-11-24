"""Assembled `editor_agent` vertical-slice module."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from ai.agents.editor.gateway.contracts import EditorGateway
from ai.agents.editor.gateway.http import HttpEditorGateway
from ai.agents.editor.github.contracts import GitHubArtifactGateway
from ai.agents.editor.tools.contracts import (
    EditorToolRegistry,
    build_default_editor_tool_registry,
)
from ai.agents.editor.workflow.contracts import EditorAgentConfig
from ai.agents.editor.workflow.graph import build_editor_agent_spec
from ai.orchestrator.contracts import AgentSpec
from ai.shared.config.agents import EditorConfig
from ai.shared.llm.factory import ChatRunnable


@dataclass(frozen=True)
class EditorModule:
    """Assembled Flux LaTeX Editor dependencies exposed for Composition Root wiring."""

    gateway: EditorGateway
    tool_registry: EditorToolRegistry
    agent_spec: AgentSpec


def build_editor_module(
    *,
    llm: ChatRunnable,
    config: EditorConfig | None = None,
    agent_config: EditorAgentConfig | None = None,
    gateway: EditorGateway | None = None,
    tool_registry: EditorToolRegistry | None = None,
    http_client: httpx.AsyncClient | None = None,
    internal_api_key: str = "",
    artifact_client: GitHubArtifactGateway | None = None,
) -> EditorModule:
    """Assemble Pages gateway adapter, tool registry, and `editor_agent` AgentSpec."""
    cfg = config or EditorConfig()
    active_gateway: EditorGateway = gateway or HttpEditorGateway(
        base_url=cfg.backend_url,
        timeout_seconds=cfg.timeout_seconds,
        http_client=http_client,
        internal_api_key=internal_api_key,
    )
    active_tools = tool_registry or build_default_editor_tool_registry(
        artifact_client=artifact_client,
    )
    eff_agent_cfg = agent_config or EditorAgentConfig(
        max_steps=cfg.max_steps,
        tool_timeout_seconds=cfg.timeout_seconds,
        timeout_seconds=min(300.0, max(45.0, cfg.timeout_seconds * cfg.max_steps)),
        max_patches_per_turn=cfg.max_patches_per_turn,
        viewport_radius_lines=cfg.viewport_radius_lines,
        max_file_chars=cfg.max_file_chars,
        hitl_policy=cfg.hitl_policy,
    )
    spec = build_editor_agent_spec(
        llm,
        active_gateway,
        tool_registry=active_tools,
        config=eff_agent_cfg,
    )
    return EditorModule(
        gateway=active_gateway,
        tool_registry=active_tools,
        agent_spec=spec,
    )


__all__ = ["EditorModule", "build_editor_module"]
