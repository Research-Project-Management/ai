"""Configuration and HITL policy type for the `library_agent` workflow."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

HitlPolicy = Literal["destructive", "all_writes", "readonly"]


class LibraryAgentConfig(BaseModel):
    """Immutable bounded execution and safety configuration for `library_agent`."""

    model_config = ConfigDict(frozen=True)

    max_steps: int = Field(default=5, ge=1, le=12)
    timeout_seconds: float = Field(default=60.0, gt=0.0, le=300.0)
    tool_timeout_seconds: float = Field(default=15.0, gt=0.0, le=120.0)
    hitl_policy: HitlPolicy = "destructive"

