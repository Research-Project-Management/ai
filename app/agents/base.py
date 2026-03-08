"""Agent base class and shared state types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ── Agent State (shared across graph nodes) ──────────────────────

class AgentState(TypedDict, total=False):
    """State that flows through the LangGraph orchestrator."""

    # Conversation messages (LangChain message objects)
    messages: Annotated[list[BaseMessage], add_messages]

    # Which agent should handle this request
    intent: str

    # Retrieved RAG context (documents / chunks)
    context: list[dict[str, Any]]

    # Final structured output from the selected agent
    agent_output: dict[str, Any]

    # Extra metadata (project_id, document_ids, etc.)
    metadata: dict[str, Any]


# ── Base Agent ───────────────────────────────────────────────────

class BaseAgent(ABC):
    """
    Abstract base for every specialized agent.

    Subclass this, implement ``run()``, and decorate with ``@register_agent``
    to add a new capability to the system.
    """

    name: str = "base"
    description: str = "Base agent"

    @abstractmethod
    async def run(self, state: AgentState) -> AgentState:
        """
        Process the current state and return an updated state.

        At minimum, the agent should populate ``state["agent_output"]``.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
