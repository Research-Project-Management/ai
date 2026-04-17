"""
Agent Card — A2A Protocol Definition
=====================================

Defines the standard interface for agent-to-agent communication in Flux-AI.
Each sub-agent exposes an AgentCard so the Orchestrator can discover and
invoke it with the right input/output contract.

Usage::

    from app.agents.card import AgentCard, AgentTask, AgentResult

    card = AgentCard(
        name="task_agent",
        description="Handles task CRUD operations",
        skills=["list_tasks", "create_task", "update_task"],
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentCard:
    """Describes a sub-agent's capabilities for the Orchestrator.

    The Orchestrator reads these cards to know:
    - What tasks each sub-agent can handle (description, skills)
    - What input to provide (input_schema hints)
    - What output to expect
    """

    name: str
    description: str
    skills: list[str] = field(default_factory=list)

    # Soft input token limit — Orchestrator won't send more context than this
    max_context_tokens: int = 4000

    # Whether this agent requires confirmation for destructive ops
    has_destructive_ops: bool = False


@dataclass
class AgentTask:
    """Input payload passed from Orchestrator to a sub-agent."""

    # The user's original request (may be filtered/focused for this agent)
    request: str

    # Conversation history (LangChain BaseMessage list)
    messages: list[Any] = field(default_factory=list)

    # Context variables
    user_id: str = ""
    workspace_id: str = ""
    project_id: str = ""
    chat_id: str = ""

    # Optional extra metadata (e.g., specific task_id to operate on)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Output returned from a sub-agent to the Orchestrator."""

    # Final text content (for Orchestrator to synthesize)
    content: str

    # Sub-agent that produced this result
    agent_name: str

    # Tool calls made during execution
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    # Whether any errors occurred (partial result)
    has_error: bool = False

    # Structured data (optional — for programmatic use)
    data: dict[str, Any] = field(default_factory=dict)


# ── Agent Card Registry ───────────────────────────────────────────────────────

_CARD_REGISTRY: dict[str, AgentCard] = {}


def register_card(card: AgentCard) -> AgentCard:
    """Register an AgentCard so the Orchestrator can discover it."""
    _CARD_REGISTRY[card.name] = card
    return card


def get_card(name: str) -> AgentCard:
    """Get a registered AgentCard by name."""
    if name not in _CARD_REGISTRY:
        raise KeyError(f"No AgentCard registered for '{name}'. Available: {list(_CARD_REGISTRY)}")
    return _CARD_REGISTRY[name]


def list_cards() -> list[AgentCard]:
    """Return all registered AgentCards."""
    return list(_CARD_REGISTRY.values())


def build_orchestrator_catalog() -> str:
    """Build a description of all sub-agents for the Orchestrator prompt."""
    lines = []
    for card in _CARD_REGISTRY.values():
        skills_str = ", ".join(card.skills) if card.skills else "general"
        lines.append(f"- **{card.name}**: {card.description} (skills: {skills_str})")
    return "\n".join(lines)
