"""Chain of Responsibility routing policies applied after Supervisor planning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ai.orchestrator.contracts import (
    AgentOutput,
    AgentRegistry,
    AgentTaskAssignment,
    RoutingDecision,
)
from ai.shared.contracts.sources import AttachedSource, parse_attached_sources

_DOWNSTREAM_ACTION_AGENTS = frozenset({"project_agent", "editor_agent"})
_READ_PREREQ_AGENTS = frozenset({"rag_agent", "library_agent"})


@dataclass(frozen=True)
class PolicyContext:
    """Immutable turn inputs inspected by routing invariant policies."""

    query: str
    request_context: Mapping[str, Any] | None
    prior_outputs: Sequence[AgentOutput]
    registry: AgentRegistry


class RoutingPolicy(Protocol):
    """Single deterministic invariant policy transforming a parsed RoutingDecision."""

    def apply(self, decision: RoutingDecision, ctx: PolicyContext) -> RoutingDecision: ...


def _select_source_prereq_agent(
    attached: Sequence[AttachedSource],
    registry: AgentRegistry,
) -> str | None:
    needs_fulltext = any(
        src.kind not in ("library", "paper") or src.indexed or src.has_pdf or bool(src.file_id)
        for src in attached
    )
    if needs_fulltext and "rag_agent" in registry.names:
        return "rag_agent"
    if "library_agent" in registry.names:
        return "library_agent"
    if "rag_agent" in registry.names:
        return "rag_agent"
    return None


class AttachedSourcePolicy:
    """Forces round-1 source reading (`rag_agent`/`library_agent`) before downstream writes."""

    def apply(self, decision: RoutingDecision, ctx: PolicyContext) -> RoutingDecision:
        if ctx.prior_outputs:
            return decision
        attached = parse_attached_sources((ctx.request_context or {}).get("attached_sources"))
        if not attached:
            return decision

        target_agents = {t.agent for t in decision.targets}
        if target_agents & _READ_PREREQ_AGENTS:
            return decision

        is_direct = decision.mode == "direct"
        is_downstream_only = (
            decision.mode == "delegate"
            and bool(target_agents)
            and target_agents.issubset(_DOWNSTREAM_ACTION_AGENTS)
        )
        if not (is_direct or is_downstream_only):
            return decision

        prereq = _select_source_prereq_agent(attached, ctx.registry)
        if prereq is None:
            return decision

        if prereq == "rag_agent":
            titles = ", ".join(
                s.title or s.effective_source_id or "attached source" for s in attached[:5]
            )
            instruction = (
                f"Read and extract key sections, findings, and actionable details from the "
                f"attached source(s) ({titles}) for the user's request: {ctx.query}"
            )
        else:
            refs = ", ".join(
                f"{s.title or 'paper'} (item_id={s.item_id})" if s.item_id else (s.title or "paper")
                for s in attached[:5]
            )
            instruction = (
                f"Inspect the attached library item(s) via get_item ({refs}) to retrieve "
                f"metadata, abstract, and notes for the user's request: {ctx.query}"
            )

        needs_followup = decision.mode == "delegate"
        downstream = ", ".join(t.agent for t in decision.targets)
        plan_summary = decision.plan_summary or (
            f"1) {prereq} -> 2) {downstream}" if needs_followup else ""
        )
        return RoutingDecision(
            mode="delegate",
            targets=(AgentTaskAssignment(agent=prereq, instruction=instruction),),
            needs_followup=needs_followup,
            plan_summary=plan_summary,
            reasoning=f"enforced attached-source prerequisite ({prereq})",
        )


class PolicyChain:
    """Ordered pipeline applying deterministic RoutingPolicy rules sequentially."""

    def __init__(self, policies: Sequence[RoutingPolicy] = ()) -> None:
        self._policies = tuple(policies)

    def apply(self, decision: RoutingDecision, ctx: PolicyContext) -> RoutingDecision:
        current = decision
        for policy in self._policies:
            current = policy.apply(current, ctx)
        return current


DEFAULT_POLICIES = PolicyChain((AttachedSourcePolicy(),))

__all__ = [
    "DEFAULT_POLICIES",
    "AttachedSourcePolicy",
    "PolicyChain",
    "PolicyContext",
    "RoutingPolicy",
]
