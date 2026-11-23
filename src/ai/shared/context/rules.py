"""Scientific Consistency & Lineage Audit Rules.

Applies the Specification / Rule Engine Pattern to audit cross-domain research integrity,
detecting unverified empirical claims, stale metrics, and broken citation chains.
"""

from __future__ import annotations

from typing import Protocol

from ai.shared.context.contracts import AuditFinding, EntityDomain, RelationType
from ai.shared.context.graph import ProvenanceGraph


class AuditRule(Protocol):
    """Specification protocol for cross-domain scientific consistency rules."""

    rule_id: str
    name: str

    def evaluate(self, graph: ProvenanceGraph) -> list[AuditFinding]:
        """Evaluate the rule against the provenance graph and return any findings."""
        ...


class UnverifiedPageClaimRule:
    """Detects pages with empirical claims lacking proof from experiment runs."""

    rule_id = "AUDIT_001"
    name = "Unverified Page Claim"

    def evaluate(self, graph: ProvenanceGraph) -> list[AuditFinding]:
        findings: list[AuditFinding] = []
        for node in graph.get_nodes_by_domain(EntityDomain.PAGE):
            # Check if node claims empirical results but lacks an incoming PROVES edge
            has_claim = bool(
                node.attributes.get("claims_metric") or node.attributes.get("has_benchmark")
            )
            if has_claim:
                in_edges = graph.get_in_edges(node.id)
                has_proof = False
                for e in in_edges:
                    if e.relation == RelationType.PROVES:
                        src_node = graph.get_node(e.src_id)
                        if src_node is not None and src_node.domain == EntityDomain.EXPERIMENT:
                            has_proof = True
                            break
                if not has_proof:
                    findings.append(
                        AuditFinding(
                            rule_id=self.rule_id,
                            severity="warning",
                            message=(
                                f"Page '{node.label}' claims empirical benchmark results "
                                "but has no verified PROVES link from an experiment run."
                            ),
                            entity_ids=(node.id,),
                        )
                    )
        return findings


class OrphanLiteratureRule:
    """Detects research papers stored in the library that are never cited or referenced in tasks."""

    rule_id = "AUDIT_002"
    name = "Orphan Literature Reference"

    def evaluate(self, graph: ProvenanceGraph) -> list[AuditFinding]:
        findings: list[AuditFinding] = []
        for node in graph.get_nodes_by_domain(EntityDomain.LITERATURE):
            out_edges = graph.get_out_edges(node.id)
            in_edges = graph.get_in_edges(node.id)

            is_cited = any(e.relation == RelationType.CITES for e in in_edges)
            informs_task = any(e.relation == RelationType.INFORMS for e in out_edges)

            if not is_cited and not informs_task:
                findings.append(
                    AuditFinding(
                        rule_id=self.rule_id,
                        severity="info",
                        message=(
                            f"Paper '{node.label}' ({node.id}) is stored in library "
                            "but not connected to any task or cited in any page."
                        ),
                        entity_ids=(node.id,),
                    )
                )
        return findings


class StaleExperimentMetricRule:
    """Detects pages citing superseded or marked-stale experiment metrics."""

    rule_id = "AUDIT_003"
    name = "Stale Experiment Metric"

    def evaluate(self, graph: ProvenanceGraph) -> list[AuditFinding]:
        findings: list[AuditFinding] = []
        for run_node in graph.get_nodes_by_domain(EntityDomain.EXPERIMENT):
            if bool(
                run_node.attributes.get("is_stale") or run_node.attributes.get("superseded_by")
            ):
                for edge in graph.get_out_edges(run_node.id):
                    if edge.relation == RelationType.PROVES:
                        page_node = graph.get_node(edge.dst_id)
                        target_label = page_node.label if page_node else edge.dst_id
                        findings.append(
                            AuditFinding(
                                rule_id=self.rule_id,
                                severity="error",
                                message=(
                                    f"Page '{target_label}' is displaying results from "
                                    f"stale/superseded run '{run_node.label}' ({run_node.id})."
                                ),
                                entity_ids=(run_node.id, edge.dst_id),
                            )
                        )
        return findings


class ScientificAuditEngine:
    """Coordinates and executes all registered scientific consistency rules."""

    def __init__(self, rules: list[AuditRule] | None = None) -> None:
        self._rules: list[AuditRule] = (
            rules
            if rules is not None
            else [
                UnverifiedPageClaimRule(),
                OrphanLiteratureRule(),
                StaleExperimentMetricRule(),
            ]
        )

    def run_audits(self, graph: ProvenanceGraph) -> list[AuditFinding]:
        findings: list[AuditFinding] = []
        for rule in self._rules:
            findings.extend(rule.evaluate(graph))
        return findings
