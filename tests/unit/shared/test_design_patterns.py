"""Unit tests for PolicyChain (Chain of Responsibility) and SseTranslator (Strategy Dispatch)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig, RunnableLambda

from ai.agents.editor.context.audit_models import (
    AuditAccumulator,
    AuditContext,
    AuditOptions,
)
from ai.agents.editor.context.audit_rules import DEFAULT_AUDIT_RULES
from ai.agents.editor.context.structure_audit import StructureAuditor
from ai.api.sse_translator import SseTranslator
from ai.orchestrator.context_policy import SlidingWindowContextPolicy
from ai.orchestrator.context_sections import (
    DEFAULT_CONTEXT_SECTIONS,
    ContextPromptComposer,
)
from ai.orchestrator.contracts import (
    AgentRegistry,
    AgentSpec,
    AgentTaskAssignment,
    RoutePlanner,
    RoutingDecision,
)
from ai.orchestrator.nodes import RootGraphNodes
from ai.orchestrator.planner import enforce_routing_invariants
from ai.orchestrator.policies import (
    AttachedSourcePolicy,
    PolicyChain,
    PolicyContext,
)
from ai.orchestrator.state import RootState
from ai.shared.workflow.events import PATCH_PREVIEW, TOOL_COMPLETED, TOOL_STARTED


def _parse_frame(frame: str) -> tuple[str, dict[str, Any]]:
    lines = [line for line in frame.strip().splitlines() if line]
    event = lines[0].removeprefix("event: ").strip()
    payload = json.loads(lines[1].removeprefix("data: ").strip())
    return event, payload


def test_policy_chain_composes_custom_and_attached_source_policies() -> None:
    async def _noop(_s: Any, _c: Any) -> dict[str, Any]:
        return {}

    registry = AgentRegistry(
        [
            AgentSpec(name="rag_agent", description="RAG", handler=_noop),
            AgentSpec(name="project_agent", description="Projects", handler=_noop),
        ]
    )

    class _UppercaseReasoningPolicy:
        def apply(self, decision: RoutingDecision, ctx: PolicyContext) -> RoutingDecision:
            del ctx
            return decision.model_copy(update={"reasoning": decision.reasoning.upper()})

    chain = PolicyChain((AttachedSourcePolicy(), _UppercaseReasoningPolicy()))
    initial = RoutingDecision(
        mode="delegate",
        targets=(AgentTaskAssignment(agent="project_agent", instruction="create tasks"),),
        reasoning="init",
    )
    out = enforce_routing_invariants(
        initial,
        query="create 5 tasks",
        request_context={
            "attached_sources": [
                {"kind": "library", "item_id": "it-1", "title": "Paper", "has_pdf": True}
            ]
        },
        prior_outputs=(),
        registry=registry,
        policies=chain,
    )
    assert out.mode == "delegate"
    assert [t.agent for t in out.targets] == ["rag_agent"]
    assert out.needs_followup is True
    assert out.reasoning == "ENFORCED ATTACHED-SOURCE PREREQUISITE (RAG_AGENT)"


def test_sse_translator_dispatches_agents_tools_channels_and_patch_preview() -> None:
    translator = SseTranslator(specialist_names=frozenset({"editor_agent"}))

    frames = []
    frames.extend(
        translator.translate(
            {
                "event": "on_chain_start",
                "name": "supervisor",
                "metadata": {"langgraph_node": "supervisor"},
            }
        )
    )
    frames.extend(
        translator.translate(
            {
                "event": "on_chain_end",
                "name": "supervisor",
                "metadata": {"langgraph_node": "supervisor"},
                "data": {
                    "output": {
                        "route": "editor_agent",
                        "routing_decision": RoutingDecision(
                            mode="delegate",
                            targets=(
                                AgentTaskAssignment(agent="editor_agent", instruction="edit"),
                            ),
                        ),
                    }
                },
            }
        )
    )
    frames.extend(
        translator.translate(
            {
                "event": "on_custom_event",
                "name": TOOL_STARTED,
                "data": {
                    "tool": "propose_patch",
                    "agent": "editor_agent",
                    "input": {"file": "a.tex"},
                },
            }
        )
    )
    frames.extend(
        translator.translate(
            {
                "event": "on_custom_event",
                "name": TOOL_COMPLETED,
                "data": {"tool": "propose_patch", "agent": "editor_agent", "status": "ok"},
            }
        )
    )
    frames.extend(
        translator.translate(
            {
                "event": "on_custom_event",
                "name": PATCH_PREVIEW,
                "data": {"patch": {"file": "a.tex"}, "total_patches": 2, "agent": "editor_agent"},
            }
        )
    )
    frames.extend(
        translator.translate(
            {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "synthesizer"},
                "data": {"chunk": AIMessageChunk(content="<think>hmm</think>Done.")},
            }
        )
    )
    frames.extend(
        translator.translate(
            {
                "event": "on_chat_model_end",
                "metadata": {"langgraph_node": "synthesizer"},
                "data": {"output": AIMessageChunk(content="")},
            }
        )
    )

    parsed = [_parse_frame(f) for f in frames]
    events = [name for name, _ in parsed]
    assert "agent.started" in events
    assert "agent.routed" in events
    assert "tool.called" in events
    assert "tool.completed" in events
    assert PATCH_PREVIEW in events
    assert "reasoning.delta" in events
    assert "text.delta" in events
    assert translator.observed_route == "editor_agent"
    assert translator.text_emitted is True


def test_structure_auditor_runs_custom_and_default_rules() -> None:
    seen_roots: list[str | None] = []

    class _RecordRootRule:
        def run(self, ctx: AuditContext, opts: AuditOptions, acc: AuditAccumulator) -> None:
            del opts, acc
            seen_roots.append(ctx.root_file)

    auditor = StructureAuditor((*DEFAULT_AUDIT_RULES, _RecordRootRule()))
    report = auditor.audit(
        {
            "main.tex": (
                "\\documentclass{article}\n"
                "\\begin{document}\n"
                "\\section{Intro}\\label{sec:intro}\n"
                "See Section~\\ref{sec:missing} and \\toprule.\n"
                "\\end{document}\n"
            )
        }
    )
    assert seen_roots == ["main.tex"]
    assert report.root_file == "main.tex"
    assert len(report.undefined_refs) == 1
    assert report.undefined_refs[0].key == "sec:missing"
    assert [p.package for p in report.missing_packages] == ["booktabs"]


def test_context_prompt_composer_composes_custom_section() -> None:
    class _VenueSection:
        def render(self, request_context: Mapping[str, Any]) -> list[str]:
            venue = str(request_context.get("venue") or "").strip()
            return [f"- Target venue: {venue}"] if venue else []

    composer = ContextPromptComposer((*DEFAULT_CONTEXT_SECTIONS, _VenueSection()))
    rendered = composer.render({"project_id": "proj-1", "venue": "NeurIPS 2026"})
    assert "## Active Request Context" in rendered
    assert "- Active project_id: proj-1" in rendered
    assert "- Target venue: NeurIPS 2026" in rendered
    assert composer.render(None) == ""


@pytest.mark.asyncio
async def test_root_graph_nodes_turn_init_and_route_after_round() -> None:
    class _StubPlanner:
        async def plan(
            self, _state: RootState, _registry: AgentRegistry, _config: RunnableConfig
        ) -> RoutingDecision:
            return RoutingDecision(mode="direct")

    nodes = RootGraphNodes(
        llm=RunnableLambda(lambda _x: AIMessage(content="ok")),
        registry=AgentRegistry(),
        planner=cast(RoutePlanner, _StubPlanner()),
        context_policy=SlidingWindowContextPolicy(),
        output_guards=(),
        has_blocking_guards=False,
        max_rounds=3,
    )

    init_state = cast(RootState, {"messages": [], "round_count": 2})
    reset = await nodes.turn_init(init_state, cast(RunnableConfig, {}))
    assert reset["round_count"] == 0
    assert reset["final_response"] == ""

    followup_state = cast(
        RootState,
        {
            "round_count": 1,
            "routing_decision": RoutingDecision(mode="delegate", needs_followup=True),
        },
    )
    assert nodes.route_after_round(followup_state) == "supervisor"
    capped_state = cast(
        RootState,
        {
            "round_count": 3,
            "routing_decision": RoutingDecision(mode="delegate", needs_followup=True),
        },
    )
    assert nodes.route_after_round(capped_state) == "synthesizer"
