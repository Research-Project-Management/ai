"""Human-in-the-loop risk policy and resume-payload parsing shared by all tool-using agents.

Risk tiers across agents: `read` (no side effects), `propose` (reviewable diff, no confirmation),
and any write tier (`write`, `safe_write`, `destructive`). Policies: `readonly`, `destructive`
(confirm only destructive tiers) and `all_writes` (confirm every write tier).
"""

from typing import Any

_NO_CONFIRM_TIERS = frozenset({"read", "propose"})
_APPROVE_WORDS = frozenset({"approve", "approved", "yes", "true"})


def is_blocked_by_policy(risk_tier: str, policy: str) -> bool:
    """Return True when the active policy forbids executing tools of this risk tier."""
    return policy == "readonly" and risk_tier != "read"


def requires_confirmation(risk_tier: str, policy: str) -> bool:
    """Return True when executing tools of this risk tier must interrupt for user approval."""
    if policy == "all_writes":
        return risk_tier not in _NO_CONFIRM_TIERS
    if policy == "destructive":
        return risk_tier == "destructive"
    return False


def parse_hitl_approval(decision: Any, interrupt_id: str | None = None) -> bool:
    """Normalize HITL resume payloads (`bool`, strings, dicts, FE `HitlDecision[]`) into bool."""
    if isinstance(decision, bool):
        return decision
    if isinstance(decision, str):
        return decision.strip().lower() in _APPROVE_WORDS
    if isinstance(decision, dict):
        if "decisions" in decision and isinstance(decision["decisions"], list | tuple):
            return parse_hitl_approval(decision["decisions"], interrupt_id=interrupt_id)
        if "approved" in decision:
            return bool(decision["approved"])
        if "decision" in decision:
            return parse_hitl_approval(decision["decision"])
        if "type" in decision:
            return parse_hitl_approval(decision["type"])
        return False
    if isinstance(decision, list | tuple):
        if not decision:
            return False
        if interrupt_id:
            for item in decision:
                if isinstance(item, dict) and (
                    item.get("interruptId") == interrupt_id or item.get("id") == interrupt_id
                ):
                    return parse_hitl_approval(item)
        if len(decision) == 1:
            return parse_hitl_approval(decision[0], interrupt_id=None)
        return all(parse_hitl_approval(item, interrupt_id=None) for item in decision)
    return False
