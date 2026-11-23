"""Configuration and HITL policy type for the `project_agent` workflow."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ProjectHitlPolicy = Literal["destructive", "all_writes", "readonly"]


class ProjectAgentConfig(BaseModel):
    """Immutable bounded execution & HITL safety configuration for `project_agent`."""

    model_config = ConfigDict(frozen=True)

    max_steps: int = Field(default=6, ge=1, le=12)
    timeout_seconds: float = Field(default=90.0, gt=0.0, le=300.0)
    tool_timeout_seconds: float = Field(default=15.0, gt=0.0, le=120.0)
    hitl_policy: ProjectHitlPolicy = "destructive"
