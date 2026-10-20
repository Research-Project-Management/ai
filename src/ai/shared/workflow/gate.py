"""Shared policy gate: validate planned calls, enforce HITL policy, and collect approvals."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from langgraph.types import interrupt
from pydantic import BaseModel, ValidationError

from ai.shared.contracts.hitl import (
    is_blocked_by_policy,
    parse_hitl_approval,
    requires_confirmation,
)


class GatedToolSpec(Protocol):
    """Structural view of a tool spec the gate needs (satisfied by every agent's spec)."""

    @property
    def name(self) -> str: ...
    @property
    def risk_tier(self) -> str: ...
    @property
    def args_schema(self) -> type[BaseModel]: ...


ConfirmationBuilder = Callable[[Any, str, dict[str, Any]], dict[str, Any]]


def run_policy_gate(
    *,
    pending: Sequence[Mapping[str, Any]],
    observations: Sequence[dict[str, Any]],
    get_spec: Callable[[str], Any],
    policy: str,
    step: int,
    id_prefix: str,
    build_confirmation: ConfirmationBuilder,
) -> dict[str, Any]:
    """Return the `policy_gate` state delta for one batch of planned calls.

    `build_confirmation(spec, interrupt_id, validated_args)` renders the HITL payload, so each
    agent keeps its own confirmation wording while sharing validation and approval flow.
    """
    approved: list[dict[str, Any]] = []
    new_observations: list[dict[str, Any]] = []
    has_rejection = False

    for idx, call in enumerate(pending):
        tool_name = str(call.get("tool") or "").strip()
        raw_args = call.get("args") if isinstance(call.get("args"), dict) else {}
        spec: GatedToolSpec | None = get_spec(tool_name)
        if spec is None:
            new_observations.append(
                {
                    "tool": tool_name or "unknown",
                    "status": "error",
                    "error": f"Unknown tool '{tool_name}'. Use only tools from Available Tools.",
                }
            )
            continue

        try:
            validated = spec.args_schema.model_validate(raw_args)
        except ValidationError as exc:
            new_observations.append(
                {
                    "tool": spec.name,
                    "status": "error",
                    "error": f"Invalid arguments: {exc.errors()[0].get('msg', str(exc))}",
                }
            )
            continue

        if is_blocked_by_policy(spec.risk_tier, policy):
            new_observations.append(
                {
                    "tool": spec.name,
                    "status": "error",
                    "error": f"Tool '{spec.name}' is blocked under active readonly policy.",
                }
            )
            continue

        validated_dict = validated.model_dump()
        if requires_confirmation(spec.risk_tier, policy):
            interrupt_id = f"{id_prefix}-{step}-{idx}-{spec.name}"
            decision = interrupt(build_confirmation(spec, interrupt_id, validated_dict))
            if not parse_hitl_approval(decision, interrupt_id=interrupt_id):
                has_rejection = True
                new_observations.append(
                    {
                        "tool": spec.name,
                        "status": "rejected",
                        "args": validated_dict,
                        "error": "Action was rejected by the user during confirmation.",
                    }
                )
                continue

        approved.append({"tool": spec.name, "args": validated_dict})

    return {
        "pending_calls": [],
        "approved_calls": approved,
        "observations": [*observations, *new_observations],
        "done": len(approved) == 0 and has_rejection,
    }
