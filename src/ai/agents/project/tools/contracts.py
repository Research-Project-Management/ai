"""Command, Runtime caches, polymorphic HITL confirmation & OCP Registry for `project_agent`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ai.agents.project.gateway.contracts import (
    ProjectGateway,
    ProjectGatewayError,
    ProjectLabel,
    ProjectMember,
    ProjectScope,
    ProjectSummary,
    WorkflowState,
)
from ai.agents.project.resolution.resolver import resolve_project
from ai.orchestrator.contracts import CitationItem
from ai.shared.workflow.registry import ToolRegistry

ProjectRiskTier = Literal["read", "write", "destructive"]


@dataclass
class ProjectToolRuntime:
    """Mutable per-turn execution runtime holding turn-scoped entity caches and UI widgets."""

    gateway: ProjectGateway
    scope: ProjectScope
    projects: list[ProjectSummary] = field(default_factory=list)
    states_by_project: dict[str, list[WorkflowState]] = field(default_factory=dict)
    members_by_project: dict[str, list[ProjectMember]] = field(default_factory=dict)
    labels_by_project: dict[str, list[ProjectLabel]] = field(default_factory=dict)
    widgets: list[dict[str, Any]] = field(default_factory=list)

    async def get_projects(self) -> list[ProjectSummary]:
        if not self.projects:
            self.projects = await self.gateway.projects.list_projects(self.scope)
        return self.projects

    async def resolve_project_id(
        self, explicit: str | None = None
    ) -> tuple[str | None, str | None]:
        raw = (explicit or "").strip()
        if raw:
            projects = await self.get_projects()
            res = resolve_project(raw, projects)
            if res.ok and res.match is not None:
                return res.match.id, None
            if res.ambiguous:
                return None, res.error
            try:
                fetched = await self.gateway.projects.get_project(self.scope, raw)
                if fetched.id and all(p.id != fetched.id for p in self.projects):
                    self.projects.append(fetched)
                return fetched.id, None
            except ProjectGatewayError:
                return None, res.error or f"Project '{raw}' was not found."

        if self.scope.project_id and self.scope.project_id.strip():
            return self.scope.project_id.strip(), None

        projects = await self.get_projects()
        if len(projects) == 1:
            return projects[0].id, None
        if not projects:
            return None, "No accessible projects found for the current user."
        choices = ", ".join(f"'{p.name}'" for p in projects[:8])
        return None, (
            f"Multiple projects available ({choices}). Specify `project_id` or project name."
        )

    async def get_states(self, project_id: str) -> list[WorkflowState]:
        if project_id not in self.states_by_project:
            self.states_by_project[project_id] = await self.gateway.projects.list_workflow_states(
                self.scope, project_id
            )
        return self.states_by_project[project_id]

    async def get_members(self, project_id: str) -> list[ProjectMember]:
        if project_id not in self.members_by_project:
            self.members_by_project[project_id] = await self.gateway.projects.list_members(
                self.scope, project_id, limit=100
            )
        return self.members_by_project[project_id]

    async def get_labels(self, project_id: str) -> list[ProjectLabel]:
        if project_id not in self.labels_by_project:
            self.labels_by_project[project_id] = await self.gateway.projects.list_labels(
                self.scope, project_id
            )
        return self.labels_by_project[project_id]

    def add_widget(self, widget: dict[str, Any]) -> None:
        w_key = (widget.get("type"), widget.get("title"))
        for idx, existing in enumerate(self.widgets):
            if (existing.get("type"), existing.get("title")) == w_key:
                self.widgets[idx] = widget
                return
        self.widgets.append(widget)


class ProjectToolResult(BaseModel):
    """Normalized execution output returned by every Project tool."""

    model_config = ConfigDict(frozen=True)

    tool: str
    status: Literal["ok", "error", "rejected"] = "ok"
    data: Any = None
    error: str | None = None
    citations: tuple[CitationItem, ...] = ()


class ProjectToolSpec(BaseModel):
    """Command specification pairing schema, risk tier, handler & polymorphic HITL text."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    args_schema: type[BaseModel]
    risk_tier: ProjectRiskTier
    idempotent: bool
    handler: Any
    describe_confirmation: Callable[[dict[str, Any]], str] | None = None

    def format_confirmation(self, validated_args: dict[str, Any]) -> str:
        """Build human-readable confirmation text owned polymorphically by each tool."""
        if self.describe_confirmation is not None:
            return self.describe_confirmation(validated_args)
        return f"Confirm executing project action `{self.name}`"


class ProjectToolRegistry(ToolRegistry[ProjectToolSpec]):
    """OCP registry of project tools used for planner formatting and policy gates."""

    label = "project"
