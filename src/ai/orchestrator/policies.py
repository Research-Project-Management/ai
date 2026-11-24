"""Chain of Responsibility routing policies applied after Supervisor planning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

from ai.orchestrator.contracts import (
    AgentName,
    AgentOutput,
    AgentRegistry,
    AgentTaskAssignment,
    RoutingDecision,
)
from ai.shared.contracts.auth import get_delegation_token
from ai.shared.contracts.sources import AttachedSource, parse_attached_sources

_DOWNSTREAM_ACTION_AGENTS = frozenset({AgentName.PROJECT, AgentName.EDITOR})
_READ_PREREQ_AGENTS = frozenset({AgentName.RAG, AgentName.LIBRARY})
_DELEGATION_REQUIRED_AGENTS: Final[frozenset[str]] = frozenset(
    {AgentName.PROJECT, AgentName.LIBRARY}
)


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
    if needs_fulltext and AgentName.RAG in registry.names:
        return AgentName.RAG
    if AgentName.LIBRARY in registry.names:
        return AgentName.LIBRARY
    if AgentName.RAG in registry.names:
        return AgentName.RAG
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

        is_downstream_only = (
            decision.mode == "delegate"
            and bool(target_agents)
            and target_agents.issubset(_DOWNSTREAM_ACTION_AGENTS)
        )
        if not is_downstream_only:
            return decision

        prereq = _select_source_prereq_agent(attached, ctx.registry)
        if prereq is None:
            return decision

        if prereq == AgentName.RAG:
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

        downstream = ", ".join(t.agent for t in decision.targets)
        plan_summary = decision.plan_summary or f"1) {prereq} -> 2) {downstream}"
        return RoutingDecision(
            mode="delegate",
            targets=(AgentTaskAssignment(agent=prereq, instruction=instruction),),
            needs_followup=True,
            plan_summary=plan_summary,
            reasoning=f"enforced attached-source prerequisite ({prereq})",
        )


class DelegationTokenPolicy:
    """Drops backend-auth-gated specialists when no user delegation token is active."""

    def apply(self, decision: RoutingDecision, ctx: PolicyContext) -> RoutingDecision:
        if decision.mode != "delegate" or not decision.targets or get_delegation_token():
            return decision
        if not any(target.agent in _DELEGATION_REQUIRED_AGENTS for target in decision.targets):
            return decision

        remaining = tuple(
            target for target in decision.targets if target.agent not in _DELEGATION_REQUIRED_AGENTS
        )
        if remaining:
            return decision.model_copy(update={"targets": remaining})
        if AgentName.RAG in ctx.registry.names:
            fallback_instruction = ctx.query or "Answer using space knowledge"
            return decision.model_copy(
                update={
                    "targets": (
                        AgentTaskAssignment(agent=AgentName.RAG, instruction=fallback_instruction),
                    ),
                    "needs_followup": False,
                }
            )
        return RoutingDecision(
            mode="direct",
            reasoning="delegation token missing for specialist targets",
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


DEFAULT_POLICIES = PolicyChain((AttachedSourcePolicy(), DelegationTokenPolicy()))

__all__ = [
    "DEFAULT_POLICIES",
    "AttachedSourcePolicy",
    "DelegationTokenPolicy",
    "PolicyChain",
    "PolicyContext",
    "RoutingPolicy",
]
