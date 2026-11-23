"""Comprehensive unit tests for the Research Context & Provenance Engine.

Verifies the Facade, Strategy, Graph, Builder, Policy, and Rule Engine patterns.
"""

from __future__ import annotations

import pytest

from ai.shared.context.budget import TokenBudgetPolicy
from ai.shared.context.builder import PromptEnvelopeBuilder
from ai.shared.context.contracts import (
    EntityDomain,
    GraphNode,
    RelationType,
    ViewportFocus,
    extract_viewport_focus,
)
from ai.shared.context.facade import ResearchContextFacade
from ai.shared.context.graph import ProvenanceGraph
from ai.shared.context.projector import (
    ContextProjector,
    EditorProjectionStrategy,
    GlobalChatProjectionStrategy,
    KanbanProjectionStrategy,
    LibraryProjectionStrategy,
)
from ai.shared.context.rules import (
    OrphanLiteratureRule,
    ScientificAuditEngine,
    StaleExperimentMetricRule,
    UnverifiedPageClaimRule,
)


def _build_test_graph() -> ProvenanceGraph:
    """Helper creating a sample cross-domain research graph."""
    graph = ProvenanceGraph()

    # 1. Literature
    paper = GraphNode(
        id="paper-vaswani",
        domain=EntityDomain.LITERATURE,
        label="Attention Is All You Need",
        attributes={"year": 2017, "cite_key": "vaswani2017"},
    )
    graph.add_node(paper)

    # 2. Governance
    task = GraphNode(
        id="task-flux-18",
        domain=EntityDomain.GOVERNANCE,
        label="Implement Multi-Head Attention",
        attributes={"status": "in_progress", "assignee": "Minh", "kind": "task"},
    )
    graph.add_node(task)

    # 3. Experiment
    run = GraphNode(
        id="run-exp-04",
        domain=EntityDomain.EXPERIMENT,
        label="Epoch 50 Evaluation Run",
        attributes={"acc": 0.942, "loss": 0.12, "is_stale": False},
    )
    graph.add_node(run)

    # 4. Page
    page = GraphNode(
        id="page-eval",
        domain=EntityDomain.PAGE,
        label="sections/eval.tex",
        attributes={"section": "Evaluation", "claims_metric": True},
    )
    graph.add_node(page)

    # Provenance Edges:
    # Paper -> Task (INFORMS)
    # Task -> Run (TRACKS)
    # Run -> Page (PROVES)
    # Page -> Paper (CITES)
    graph.add_edge("paper-vaswani", "task-flux-18", RelationType.INFORMS)
    graph.add_edge("task-flux-18", "run-exp-04", RelationType.TRACKS)
    graph.add_edge("run-exp-04", "page-eval", RelationType.PROVES)
    graph.add_edge("page-eval", "paper-vaswani", RelationType.CITES)

    return graph


def test_provenance_graph_basic_operations() -> None:
    graph = _build_test_graph()

    assert graph.node_count == 4
    assert graph.edge_count == 4
    assert graph.get_node("task-flux-18") is not None
    assert graph.get_node("non-existent") is None

    assert graph.has_edge("paper-vaswani", "task-flux-18", RelationType.INFORMS)
    assert not graph.has_edge("paper-vaswani", "task-flux-18", RelationType.PROVES)
    assert not graph.has_edge("page-eval", "task-flux-18")


def test_provenance_graph_k_hop_traversal() -> None:
    graph = _build_test_graph()

    # 1-hop around run-exp-04 should reach task-flux-18 (in) and page-eval (out)
    one_hop = graph.get_k_hop_subgraph("run-exp-04", k=1)
    one_hop_ids = {n.id for n in one_hop}
    assert one_hop_ids == {"task-flux-18", "page-eval"}

    # 2-hop around run-exp-04 should also reach paper-vaswani
    two_hop = graph.get_k_hop_subgraph("run-exp-04", k=2)
    two_hop_ids = {n.id for n in two_hop}
    assert two_hop_ids == {"task-flux-18", "page-eval", "paper-vaswani"}

    # Domain filtering
    gov_only = graph.get_k_hop_subgraph("run-exp-04", k=2, domain_filter=[EntityDomain.GOVERNANCE])
    assert [n.id for n in gov_only] == ["task-flux-18"]


def test_provenance_graph_upstream_lineage_tracing() -> None:
    graph = _build_test_graph()

    # Upstream lineage of page-eval:
    # page-eval <-[PROVES]- run-exp-04 <-[TRACKS]- task-flux-18 <-[INFORMS]- paper-vaswani
    lineage = graph.get_upstream_lineage("page-eval")
    lineage_ids = [n.id for n in lineage]

    assert "run-exp-04" in lineage_ids
    assert "task-flux-18" in lineage_ids
    assert "paper-vaswani" in lineage_ids


def test_token_budget_policy_allocation_and_fitting() -> None:
    policy = TokenBudgetPolicy(max_context_tokens=1000)

    assert policy.primacy_budget == 200
    assert policy.middle_budget == 500
    assert policy.recency_budget == 300

    short_text = "Hello research world"
    assert policy.fit_text(short_text, max_tokens=50) == short_text

    long_text = "Line 1\n" + "Very long analysis line\n" * 50
    fitted = policy.fit_text(long_text, max_tokens=20)
    assert "[truncated to budget]" in fitted

    # Invalid configurations
    with pytest.raises(ValueError, match="max_context_tokens must be >= 100"):
        TokenBudgetPolicy(max_context_tokens=50)

    with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
        TokenBudgetPolicy(primacy_ratio=0.5, middle_ratio=0.5, recency_ratio=0.5)


def test_context_projector_strategies() -> None:
    projector = ContextProjector()

    editor_strat = projector.resolve(ViewportFocus(view="editor"))
    assert isinstance(editor_strat, EditorProjectionStrategy)
    assert editor_strat.get_domain_weights()[EntityDomain.PAGE] > 0.50

    kanban_strat = projector.resolve(ViewportFocus(view="kanban"))
    assert isinstance(kanban_strat, KanbanProjectionStrategy)
    assert kanban_strat.get_domain_weights()[EntityDomain.GOVERNANCE] > 0.50

    lib_strat = projector.resolve(ViewportFocus(view="library"))
    assert isinstance(lib_strat, LibraryProjectionStrategy)
    assert lib_strat.get_domain_weights()[EntityDomain.LITERATURE] > 0.50

    chat_strat = projector.resolve(ViewportFocus(view="fullscreen_chat"))
    assert isinstance(chat_strat, GlobalChatProjectionStrategy)


def test_prompt_envelope_builder_structure() -> None:
    builder = PromptEnvelopeBuilder()
    builder.with_research_identity(
        "FluxKV Compression",
        venue="NeurIPS 2026",
        deadline="2026-12-15",
        hypothesis="KV compression cuts memory by 40%",
    )
    builder.with_governance_summary(
        milestones=["M1: Prototype", "M2: Eval"],
        active_tasks=["FLUX-10: Baseline", "FLUX-12: Training"],
    )
    node = GraphNode(
        id="exp-1",
        domain=EntityDomain.EXPERIMENT,
        label="Run 1",
        attributes={"acc": 0.95},
    )
    builder.with_provenance_evidence([node])
    builder.with_active_focus(
        ViewportFocus(view="editor", selection="\\section{Method}"),
        query="Write the next subsection",
    )

    envelope = builder.build()

    assert "FluxKV Compression" in envelope.primacy_text
    assert "NeurIPS 2026" in envelope.primacy_text
    assert "Run 1" in envelope.middle_text
    assert "\\section{Method}" in envelope.recency_text
    assert "Write the next subsection" in envelope.recency_text
    assert envelope.total_tokens > 0

    full = envelope.full_prompt
    assert (
        full.index("FluxKV Compression")
        < full.index("Run 1")
        < full.index("Write the next subsection")
    )


def test_scientific_audit_rules() -> None:
    engine = ScientificAuditEngine()
    graph = ProvenanceGraph()

    # 1. Unverified Page Claim: Page claims metric but no PROVES edge
    unverified_page = GraphNode(
        id="page-unverified",
        domain=EntityDomain.PAGE,
        label="sections/results.tex",
        attributes={"claims_metric": True},
    )
    graph.add_node(unverified_page)

    # 2. Orphan Literature: Paper not linked to anything
    orphan_paper = GraphNode(
        id="paper-orphan",
        domain=EntityDomain.LITERATURE,
        label="Unused Paper 2024",
    )
    graph.add_node(orphan_paper)

    findings = engine.run_audits(graph)
    rule_ids = {f.rule_id for f in findings}

    assert UnverifiedPageClaimRule.rule_id in rule_ids
    assert OrphanLiteratureRule.rule_id in rule_ids

    # 3. Stale Experiment Metric
    stale_run = GraphNode(
        id="run-stale",
        domain=EntityDomain.EXPERIMENT,
        label="Old Run",
        attributes={"is_stale": True},
    )
    graph.add_node(stale_run)
    graph.add_edge("run-stale", "page-unverified", RelationType.PROVES)

    stale_findings = engine.run_audits(graph)
    stale_rule_ids = {f.rule_id for f in stale_findings}
    assert StaleExperimentMetricRule.rule_id in stale_rule_ids


def test_research_context_facade_end_to_end() -> None:
    graph = _build_test_graph()
    facade = ResearchContextFacade(graph=graph)

    # Assemble context from Editor viewpoint focusing on page-eval
    focus = ViewportFocus(
        view="editor",
        target_id="page-eval",
        selection="We achieve 94.2% accuracy.",
    )
    envelope = facade.assemble_context(
        focus,
        "Verify this claim against our experiments",
        research_title="Adaptive Attention",
        target_venue="ICML 2026",
    )

    assert "Adaptive Attention" in envelope.primacy_text
    assert "ICML 2026" in envelope.primacy_text
    assert "We achieve 94.2% accuracy." in envelope.recency_text

    # Upstream claim origin tracing
    origin = facade.trace_claim_origin("page-eval")
    origin_ids = {n.id for n in origin}
    assert "run-exp-04" in origin_ids
    assert "task-flux-18" in origin_ids
    assert "paper-vaswani" in origin_ids

    # Summary metrics
    metrics = facade.get_summary_metrics()
    assert metrics["total_nodes"] == 4
    assert metrics["total_edges"] == 4
    assert metrics["domain_counts"]["gov"] == 1
    assert metrics["domain_counts"]["exp"] == 1


def test_extract_viewport_focus_all_views() -> None:
    # 1. Empty context defaults to fullscreen_chat
    assert extract_viewport_focus(None).view == "fullscreen_chat"
    assert extract_viewport_focus({}).view == "fullscreen_chat"

    # 2. Editor view inferred from active_file_path or page_id
    ed_focus = extract_viewport_focus(
        {"active_file_path": "sections/intro.tex", "selection": "text"}
    )
    assert ed_focus.view == "editor"
    assert ed_focus.target_id == "sections/intro.tex"
    assert ed_focus.selection == "text"

    # 3. Kanban view inferred from project_id without page_id
    kb_focus = extract_viewport_focus({"project_id": "proj-123"})
    assert kb_focus.view == "kanban"
    assert kb_focus.target_id == "proj-123"

    # 4. Library view inferred from attached_sources
    lib_focus = extract_viewport_focus({"attached_sources": [{"item_id": "item-abc"}]})
    assert lib_focus.view == "library"
    assert lib_focus.target_id == "item-abc"


@pytest.mark.asyncio
async def test_supervisor_planner_with_context_facade_integration() -> None:
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.runnables import RunnableLambda

    from ai.orchestrator.contracts import AgentRegistry, AgentSpec
    from ai.orchestrator.planner import LLMSupervisorPlanner
    from ai.shared.llm.factory import ChatRunnable

    captured_prompts: list[str] = []

    def _mock_router(messages: LanguageModelInput) -> AIMessage:
        if isinstance(messages, list):
            for m in messages:
                content = getattr(m, "content", "")
                if isinstance(content, str):
                    captured_prompts.append(content)
        return AIMessage(content='{"mode": "direct", "reasoning": "done"}')

    router_llm: ChatRunnable = RunnableLambda(_mock_router)
    graph = _build_test_graph()
    facade = ResearchContextFacade(graph=graph)

    planner = LLMSupervisorPlanner(router_llm, context_facade=facade)

    state = {
        "messages": [HumanMessage(content="Check if baseline matches Vaswani")],
        "context": {
            "project_title": "Adaptive Attention Research",
            "target_venue": "NeurIPS 2026",
            "active_file_path": "sections/eval.tex",
            "selection": "Table 2 results",
        },
    }
    registry = AgentRegistry(
        [AgentSpec(name="editor_agent", description="LaTeX editor", handler=lambda s, c: "ok")]
    )

    decision = await planner.plan(state, registry, {})  # type: ignore[arg-type]
    assert decision.mode == "direct"
    assert len(captured_prompts) > 0

    full_system_prompt = captured_prompts[0]
    assert "Adaptive Attention Research" in full_system_prompt
    assert "Target Venue: NeurIPS 2026" in full_system_prompt
    assert "sections/eval.tex" in full_system_prompt
    assert "Attention Is All You Need" in full_system_prompt
