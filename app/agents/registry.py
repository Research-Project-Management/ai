"""Agent registry — auto-discovery via @register_agent decorator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.agents.base import BaseAgent

# Global registry mapping agent name → agent instance
_REGISTRY: dict[str, BaseAgent] = {}


def register_agent(cls: type[BaseAgent]) -> type[BaseAgent]:
    """
    Class decorator that instantiates the agent and registers it.

    Usage::

        @register_agent
        class ChatAgent(BaseAgent):
            name = "chat"
            description = "General conversational AI"
            ...
    """
    instance = cls()
    _REGISTRY[instance.name] = instance
    # Invalidate supervisor's agent catalog cache so new agents are immediately
    # visible in the Supervisor prompt without a server restart.
    try:
        from app.agents.graph import invalidate_agent_catalog_cache
        invalidate_agent_catalog_cache()
    except Exception:
        pass  # graph may not be imported yet during initial startup — that's fine
    return cls


def get_agent(name: str) -> BaseAgent:
    """Return a registered agent by name, or raise KeyError."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Agent '{name}' not found. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_agents() -> dict[str, BaseAgent]:
    """Return the full registry (name → instance)."""
    return dict(_REGISTRY)


def agent_descriptions() -> list[dict[str, str]]:
    """
    Return a list of ``{name, description}`` for every registered agent.

    Used by the intent classifier to decide which agent to route to.
    """
    return [
        {"name": a.name, "description": a.description}
        for a in _REGISTRY.values()
    ]
