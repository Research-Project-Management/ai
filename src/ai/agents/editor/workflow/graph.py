"""Subgraph compiler and Root Orchestrator AgentSpec adapter for `editor_agent`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from ai.agents.editor.gateway.contracts import EditorGateway
from ai.agents.editor.tools.contracts import (
    EditorToolRegistry,
    build_default_editor_tool_registry,
)
from ai.agents.editor.workflow.contracts import EditorAgentConfig
from ai.agents.editor.workflow.nodes import build_editor_nodes
from ai.agents.editor.workflow.state import EditorState
from ai.orchestrator.contracts import AgentSpec
from ai.orchestrator.state import RootState
from ai.shared.llm.factory import ChatRunnable
from ai.shared.workflow.graph import build_tool_agent_spec, compile_tool_agent_graph
from ai.shared.workflow.turn import extract_peer_findings


def build_editor_subgraph(
    llm: ChatRunnable,
    gateway: EditorGateway,
    *,
    tool_registry: EditorToolRegistry | None = None,
    config: EditorAgentConfig | None = None,
) -> CompiledStateGraph:
    """Compile `hydrate_context -> plan_and_act -> policy_gate -> execute_and_verify -> synthesize_response`."""
    cfg = config or EditorAgentConfig()
    registry = tool_registry or build_default_editor_tool_registry()
    nodes = build_editor_nodes(
        llm=llm,
        gateway=gateway,
        tool_registry=registry,
        agent_config=cfg,
    )

    return compile_tool_agent_graph(EditorState, nodes, max_steps=cfg.max_steps)


def _extract_provenance_artifacts(observations: Sequence[Any]) -> list[dict[str, str]]:
    """Collect verified GitHub experiment artifact provenance from successful table insertions."""
    items: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        if obs.get("tool") != "insert_experiment_table" or obs.get("status") != "ok":
            continue
        data = obs.get("data")
        if not isinstance(data, dict):
            continue
        artifact = data.get("artifact")
        if not isinstance(artifact, dict):
            continue
        repo = str(artifact.get("repo") or "").strip()
        path = str(artifact.get("path") or "").strip()
        commit_sha = str(artifact.get("commit_sha") or "").strip()
        url = str(artifact.get("url") or "").strip()
        if not (repo and path and commit_sha and url):
            continue
        key = (repo, path, commit_sha)
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "repo": repo,
                "path": path,
                "commit_sha": commit_sha,
                "url": url,
            }
        )
    return items


def build_editor_agent_spec(
    llm: ChatRunnable,
    gateway: EditorGateway,
    *,
    tool_registry: EditorToolRegistry | None = None,
    config: EditorAgentConfig | None = None,
) -> AgentSpec:
    """Build the pluggable `editor_agent` AgentSpec with anchor-verified LaTeX patching."""
    cfg = config or EditorAgentConfig()
    subgraph = build_editor_subgraph(
        llm,
        gateway,
        tool_registry=tool_registry,
        config=cfg,
    )

    def _extra_state(
        state: RootState, context: dict[str, Any], configurable: dict[str, Any]
    ) -> dict[str, Any]:
        page_id = str(configurable.get("page_id") or context.get("page_id") or "").strip()
        raw_ctx = {
            **context,
            **({"page_id": page_id} if page_id else {}),
        }
        return {
            "peer_findings": extract_peer_findings(
                state, exclude_agent="editor_agent", include_citations=True
            ),
            "page_id": page_id,
            "raw_editor_context": raw_ctx,
            "patches": [],
        }

    def _extra_artifacts(result: Mapping[str, Any]) -> dict[str, Any]:
        provenance = _extract_provenance_artifacts(result.get("observations") or ())
        return {
            "editor_patches": list(result.get("patches") or ()),
            **({"provenance": provenance} if provenance else {}),
        }

    return build_tool_agent_spec(
        name="editor_agent",
        description=(
            "Creates new LaTeX pages/articles (`create_page`) and reads multi-file LaTeX/.bib pages, "
            "understands cursor/selection/compile diagnostics, explains mathematical or LaTeX sections, "
            "checks citations/labels/lint rules, and proposes anchor-verified LaTeX diff patches for "
            "interactive CodeMirror review."
        ),
        capabilities=(
            "page_creation",
            "latex_editing",
            "latex_debugging",
            "page_navigation",
            "patch_proposal",
            "citation_inspection",
        ),
        timeout_seconds=cfg.timeout_seconds,
        subgraph=subgraph,
        empty_query_message="Yêu cầu chỉnh sửa LaTeX đang trống.",
        build_extra_state=_extra_state,
        build_extra_artifacts=_extra_artifacts,
    )
