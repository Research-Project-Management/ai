"""Boundary-Anchored Prompt Envelope Builder.

Applies the Builder Pattern to construct structured, 3-zone prompt envelopes
that prevent the 'Lost in the Middle' degradation effect (Stanford TACL 2024).
"""

from __future__ import annotations

from collections.abc import Sequence

from ai.shared.context.budget import TokenBudgetPolicy
from ai.shared.context.contracts import ContextEnvelope, GraphNode, ViewportFocus


class PromptEnvelopeBuilder:
    """Fluent builder for three-zone research context envelopes."""

    def __init__(self, budget_policy: TokenBudgetPolicy | None = None) -> None:
        self._policy = budget_policy or TokenBudgetPolicy()
        self._primacy_sections: list[str] = []
        self._middle_sections: list[str] = []
        self._recency_sections: list[str] = []
        self._included_nodes: list[GraphNode] = []

    def with_research_identity(
        self,
        title: str,
        *,
        venue: str | None = None,
        deadline: str | None = None,
        hypothesis: str | None = None,
    ) -> PromptEnvelopeBuilder:
        """Add research metadata to the high-attention Primacy zone."""
        lines = [f"# Research Project: {title.strip()}"]
        if venue:
            lines.append(f"Target Venue: {venue.strip()}")
        if deadline:
            lines.append(f"Submission Deadline: {deadline.strip()}")
        if hypothesis:
            lines.append(f"Core Hypothesis: {hypothesis.strip()}")
        self._primacy_sections.append("\n".join(lines))
        return self

    def with_governance_summary(
        self,
        *,
        milestones: Sequence[str] = (),
        active_tasks: Sequence[str] = (),
    ) -> PromptEnvelopeBuilder:
        """Add governance highlights to the Primacy zone."""
        lines: list[str] = []
        if milestones:
            lines.append("Active Milestones: " + "; ".join(milestones[:3]))
        if active_tasks:
            lines.append("Critical Tasks: " + "; ".join(active_tasks[:5]))
        if lines:
            self._primacy_sections.append("## Governance Snapshot\n" + "\n".join(lines))
        return self

    def with_provenance_evidence(
        self,
        nodes: Sequence[GraphNode],
    ) -> PromptEnvelopeBuilder:
        """Add linked scientific entities to the Middle supplementary zone."""
        self._included_nodes.extend(nodes)
        lines: list[str] = ["## Interconnected Research Evidence"]
        for node in nodes:
            attr_summary = ", ".join(f"{k}={v}" for k, v in list(node.attributes.items())[:3])
            lines.append(f"- [{node.domain.upper()}] {node.label} ({node.id}) {attr_summary}")
        if len(lines) > 1:
            self._middle_sections.append("\n".join(lines))
        return self

    def with_active_focus(
        self,
        focus: ViewportFocus,
        query: str,
    ) -> PromptEnvelopeBuilder:
        """Add active selection and user query to the high-attention Recency zone."""
        lines: list[str] = []
        if focus.selection and focus.selection.strip():
            lines.append(f"### Active Selection ({focus.view}):\n{focus.selection.strip()}")
        if focus.attached_sources:
            lines.append("Attached Source IDs: " + ", ".join(focus.attached_sources))
        lines.append(f"### Researcher Request:\n{query.strip()}")
        self._recency_sections.append("\n".join(lines))
        return self

    def build(self) -> ContextEnvelope:
        """Assemble and enforce token budget constraints across all three zones."""
        raw_primacy = "\n\n".join(self._primacy_sections)
        raw_middle = "\n\n".join(self._middle_sections)
        raw_recency = "\n\n".join(self._recency_sections)

        primacy_fitted = self._policy.fit_text(raw_primacy, self._policy.primacy_budget)
        middle_fitted = self._policy.fit_text(raw_middle, self._policy.middle_budget)
        recency_fitted = self._policy.fit_text(raw_recency, self._policy.recency_budget)

        total_tokens = (
            self._policy.estimate_tokens(primacy_fitted)
            + self._policy.estimate_tokens(middle_fitted)
            + self._policy.estimate_tokens(recency_fitted)
        )

        return ContextEnvelope(
            primacy_text=primacy_fitted,
            middle_text=middle_fitted,
            recency_text=recency_fitted,
            total_tokens=total_tokens,
            included_nodes=tuple(self._included_nodes),
        )
