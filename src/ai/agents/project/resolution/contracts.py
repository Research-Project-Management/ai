"""Immutable contracts for natural-language entity resolution and turn context."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ai.agents.project.gateway.contracts import (
    ProjectLabel,
    ProjectMember,
    ProjectSummary,
    WorkflowState,
)


class CandidateItem(BaseModel):
    """Normalized candidate entity fed into the resolution strategy chain."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    aliases: tuple[str, ...] = ()


class ResolvedEntity(BaseModel):
    """Successful unambiguous match returned by `resolve_entity`."""

    model_config = ConfigDict(frozen=True)

    id: str
    label: str
    strategy: str


class ResolutionResult(BaseModel):
    """Structured outcome distinguishing unique match, ambiguity, or not-found."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    match: ResolvedEntity | None = None
    ambiguous: tuple[CandidateItem, ...] = ()
    error: str | None = None


class ProjectContextSnapshot(BaseModel):
    """Hydrated per-turn snapshot of active project metadata, states, members & labels."""

    model_config = ConfigDict(frozen=True)

    active_project: ProjectSummary | None = None
    projects: tuple[ProjectSummary, ...] = ()
    workflow_states: tuple[WorkflowState, ...] = ()
    members: tuple[ProjectMember, ...] = ()
    labels: tuple[ProjectLabel, ...] = ()
    user_id: str = ""
    space_id: str = ""
