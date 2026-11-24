"""Immutable contracts, registry, context policy, and guard protocols for orchestration."""

import re
from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Any, Final, Literal, Protocol

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai.shared.contracts.run_control import RunUsage


class AgentName(StrEnum):
    """Canonical registered specialist agent identifiers."""

    RAG = "rag_agent"
    LIBRARY = "library_agent"
    PROJECT = "project_agent"
    EDITOR = "editor_agent"
    WEB_SEARCH = "web_search_agent"


RESERVED_NODE_NAMES: Final[frozenset[str]] = frozenset(
    {"turn_init", "supervisor", "round_gate", "direct_respond", "synthesizer", "output_guard"}
)
_AGENT_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]{1,48}$")


class CitationItem(BaseModel):
    """Structured provenance item returned by specialist agents."""

    model_config = ConfigDict(frozen=True)

    index: int | None = Field(default=None, ge=1)
    source_id: str = Field(..., min_length=1)
    title: str = ""
    url: str | None = None
    page: int | None = None
    snippet: str = ""


class AgentOutput(BaseModel):
    """Normalized immutable result produced by a specialist agent node."""

    model_config = ConfigDict(frozen=True)

    agent: str = Field(..., min_length=1)
    content: str = ""
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
    citations: tuple[CitationItem, ...] = ()
    artifacts: dict[str, Any] = Field(default_factory=dict)
    usage: RunUsage = Field(default_factory=RunUsage)
    is_final: bool = False


class AgentSpec(BaseModel):
    """Declarative registration spec for plugging a specialist agent into the root graph."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    description: str = Field(..., min_length=1)
    handler: Any
    capabilities: tuple[str, ...] = ()
    timeout_seconds: float = Field(default=60.0, gt=0.0, le=600.0)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        stripped = value.strip()
        if stripped == "direct" or stripped in RESERVED_NODE_NAMES:
            raise ValueError(f"Agent name '{stripped}' is reserved")
        if not _AGENT_NAME_RE.match(stripped):
            raise ValueError(f"Agent name '{stripped}' must match ^[a-z][a-z0-9_]{{1,48}}$")
        return stripped

    @field_validator("handler")
    @classmethod
    def _validate_handler(cls, value: Any) -> Any:
        if not (callable(value) or hasattr(value, "ainvoke")):
            raise ValueError("Agent handler must be callable or expose .ainvoke()")
        return value


class AgentRegistry:
    """Immutable catalog of specialist agents registered in the orchestrator."""

    def __init__(self, specs: Iterable[AgentSpec] = ()) -> None:
        ordered: list[AgentSpec] = []
        by_name: dict[str, AgentSpec] = {}
        for spec in specs:
            if spec.name in by_name:
                raise ValueError(f"Duplicate agent name: {spec.name}")
            ordered.append(spec)
            by_name[spec.name] = spec
        self._specs: tuple[AgentSpec, ...] = tuple(ordered)
        self._by_name: dict[str, AgentSpec] = by_name

    @property
    def specs(self) -> tuple[AgentSpec, ...]:
        return self._specs

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._by_name.keys())

    @property
    def is_empty(self) -> bool:
        return len(self._specs) == 0

    def get(self, name: str) -> AgentSpec | None:
        return self._by_name.get(name)

    def format_catalog(self) -> str:
        return "\n".join(f"- **{s.name}**: {s.description}" for s in self._specs)


class AgentTaskAssignment(BaseModel):
    """A single delegated task targeting one registered specialist agent."""

    model_config = ConfigDict(frozen=True)

    agent: str = Field(..., min_length=1)
    instruction: str = Field(..., min_length=1)


class RoutingDecision(BaseModel):
    """Structured routing decision emitted by a RoutePlanner."""

    model_config = ConfigDict(frozen=True)

    mode: Literal["direct", "delegate", "synthesize"] = "direct"
    targets: tuple[AgentTaskAssignment, ...] = ()
    needs_followup: bool = False
    plan_summary: str = ""
    reasoning: str = ""


class RoutePlanner(Protocol):
    """Strategy interface for routing and multi-agent task planning."""

    async def plan(
        self,
        state: Any,
        registry: AgentRegistry,
        config: RunnableConfig,
    ) -> RoutingDecision: ...


class ContextWindowPolicy(Protocol):
    """Strategy interface for bounding conversation history sent to LLMs."""

    def select_messages(self, messages: Sequence[BaseMessage]) -> list[BaseMessage]: ...


class DraftOutput(BaseModel):
    """Immutable pre-delivery output passed through the OutputGuard chain."""

    model_config = ConfigDict(frozen=True)

    content: str
    citations: tuple[CitationItem, ...] = ()
    artifacts: dict[str, Any] = Field(default_factory=dict)
    agent_outputs: tuple[AgentOutput, ...] = ()


class OutputGuard(Protocol):
    """Post-generation verification or transformation step in the output pipeline."""

    async def validate(
        self,
        draft: DraftOutput,
        state: Any,
        config: RunnableConfig,
    ) -> DraftOutput: ...


class InputGuardViolation(ValueError):
    """Raised when inbound messages violate an InputGuard policy."""


class InputGuard(Protocol):
    """Pre-execution validation or sanitization policy for inbound messages."""

    async def validate_messages(
        self,
        messages: Sequence[BaseMessage],
        config: RunnableConfig,
    ) -> list[BaseMessage]: ...
