"""Command and Registry contracts for `library_agent` tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ai.orchestrator.contracts import CitationItem
from ai.shared.workflow.registry import ToolRegistry

RiskTier = Literal["read", "safe_write", "destructive"]


class ToolResult(BaseModel):
    """Normalized execution output returned by every Library tool."""

    model_config = ConfigDict(frozen=True)

    tool: str
    status: Literal["ok", "error", "rejected"] = "ok"
    data: Any = None
    error: str | None = None
    citations: tuple[CitationItem, ...] = ()


ConfirmationBuilder = Callable[[str, dict[str, Any], dict[str, str]], dict[str, Any]]


class LibraryToolSpec(BaseModel):
    """Declarative specification for one intent-oriented Library tool."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    args_schema: type[BaseModel]
    risk_tier: RiskTier
    idempotent: bool
    handler: Any
    describe_confirmation: ConfirmationBuilder | None = None

    def build_confirmation(
        self,
        *,
        interrupt_id: str,
        validated_args: dict[str, Any],
        known_titles: dict[str, str],
    ) -> dict[str, Any]:
        """Build a human-readable confirmation payload for this tool."""
        if self.describe_confirmation is not None:
            return self.describe_confirmation(interrupt_id, validated_args, known_titles)
        return {
            "id": interrupt_id,
            "tool": self.name,
            "description": f"Confirm executing library action `{self.name}`",
            "args": validated_args,
            "allowed_decisions": ["approve", "reject"],
        }


class LibraryToolRegistry(ToolRegistry[LibraryToolSpec]):
    """Registry of library tools used for planner formatting and policy gates."""

    label = "library"


def build_default_tool_registry() -> LibraryToolRegistry:
    """Assemble the default 11-tool registry for `library_agent`."""
    from ai.agents.library.tools.destructive import build_destructive_tools
    from ai.agents.library.tools.read import build_read_tools
    from ai.agents.library.tools.write import build_write_tools

    return LibraryToolRegistry(
        (
            *build_read_tools(),
            *build_write_tools(),
            *build_destructive_tools(),
        )
    )
