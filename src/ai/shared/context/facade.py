"""Unified Facade for the Research Context & Provenance Engine.

Applies the Facade Pattern to provide a clean, single entry point for Orchestrators,
Specialist Agents, and Backend Services to resolve boundary-anchored research contexts.
"""

from __future__ import annotations

from typing import Any

from ai.shared.context.budget import TokenBudgetPolicy
from ai.shared.context.builder import PromptEnvelopeBuilder
from ai.shared.context.contracts import (
    AuditFinding,
    ContextEnvelope,
    EntityDomain,
    GraphEdge,
    GraphNode,
    RelationType,
    ViewportFocus,
)
from ai.shared.context.graph import ProvenanceGraph
from ai.shared.context.projector import ContextProjector
from ai.shared.context.rules import ScientificAuditEngine


class ResearchContextFacade:
    """Unified Facade encapsulating graph traversal, projection, budgeting, and auditing."""

    def __init__(
        self,
        *,
        graph: ProvenanceGraph | None = None,
        projector: ContextProjector | None = None,
        budget_policy: TokenBudgetPolicy | None = None,
        audit_engine: ScientificAuditEngine | None = None,
    ) -> None:
        self._graph = graph or ProvenanceGraph()
        self._projector = projector or ContextProjector()
        self._budget = budget_policy or TokenBudgetPolicy()
        self._audit = audit_engine or ScientificAuditEngine()

    @property
    def graph(self) -> ProvenanceGraph:
        return self._graph

    def register_node(self, node: GraphNode) -> None:
        """Register or update an entity node in the research graph."""
        self._graph.add_node(node)

    def link_entities(
        self,
        src_id: str,
        dst_id: str,
        relation: RelationType,
        metadata: dict[str, object] | None = None,
    ) -> GraphEdge:
        """Create a directed provenance relation between two research entities."""
        return self._graph.add_edge(src_id, dst_id, relation, metadata)

    def assemble_context(
        self,
        focus: ViewportFocus,
        query: str,
        *,
        research_title: str = "Research Project",
        target_venue: str | None = None,
        deadline: str | None = None,
        hypothesis: str | None = None,
        k_hop: int = 1,
    ) -> ContextEnvelope:
        """Assemble an optimal boundary-anchored ContextEnvelope for LLM reasoning.

        1. Resolves projection strategy based on the active viewport.
        2. Retrieves relevant k-hop neighborhood from provenance graph.
        3. Enforces knapsack token quotas across domains.
        4. Fluent assembly into Primacy, Middle, and Recency zones.
        """
        strategy = self._projector.resolve(focus)
        weights = strategy.get_domain_weights()

        # 1. Determine anchor entities
        anchor_id = focus.target_id or focus.inferred_entity_id
        candidate_nodes: list[GraphNode] = []

        if anchor_id and self._graph.get_node(anchor_id):
            candidate_nodes = self._graph.get_k_hop_subgraph(anchor_id, k=k_hop)
        elif focus.attached_sources:
            for src_id in focus.attached_sources:
                candidate_nodes.extend(self._graph.get_k_hop_subgraph(src_id, k=1))
        else:
            # Macro project fallback: collect highest-priority nodes from each domain
            candidate_nodes = list(self._graph.nodes)

        # 2. Allocate token quotas according to strategy weights
        fitted_nodes = self._budget.allocate_nodes(candidate_nodes, weights)

        # 3. Build boundary-anchored prompt envelope
        builder = PromptEnvelopeBuilder(self._budget)
        builder.with_research_identity(
            research_title,
            venue=target_venue,
            deadline=deadline,
            hypothesis=hypothesis,
        )

        # Extract governance highlights if present
        gov_nodes = [n for n in fitted_nodes if n.domain == EntityDomain.GOVERNANCE]
        tasks = [n.label for n in gov_nodes if "task" in str(n.attributes.get("kind") or "task")]
        builder.with_governance_summary(active_tasks=tasks)

        builder.with_provenance_evidence(fitted_nodes)
        builder.with_active_focus(focus, query)

        return builder.build()

    def audit_integrity(self) -> list[AuditFinding]:
        """Audit the research graph for cross-domain inconsistencies or missing evidence."""
        return self._audit.run_audits(self._graph)

    def trace_claim_origin(self, target_id: str) -> list[GraphNode]:
        """Trace the backward chain of custody for a claim, table, or paper back to its origin."""
        return self._graph.get_upstream_lineage(target_id)

    def get_summary_metrics(self) -> dict[str, Any]:
        """Return quantitative metrics on the research project's knowledge graph."""
        domain_counts: dict[str, int] = {d.value: 0 for d in EntityDomain}
        for node in self._graph.nodes:
            domain_counts[node.domain.value] += 1

        findings = self.audit_integrity()
        return {
            "total_nodes": self._graph.node_count,
            "total_edges": self._graph.edge_count,
            "domain_counts": domain_counts,
            "audit_warnings": len([f for f in findings if f.severity == "warning"]),
            "audit_errors": len([f for f in findings if f.severity == "error"]),
        }
