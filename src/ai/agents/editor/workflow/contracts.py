"""Configuration and HITL policy type for the `editor_agent` workflow."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EditorHitlPolicy = Literal["all_writes", "readonly"]


class EditorAgentConfig(BaseModel):
    """Immutable bounded execution and safety configuration for `editor_agent`."""

    model_config = ConfigDict(frozen=True)

    max_steps: int = Field(default=6, ge=1, le=12)
    timeout_seconds: float = Field(default=90.0, gt=0.0, le=300.0)
    tool_timeout_seconds: float = Field(default=15.0, gt=0.0, le=120.0)
    max_patches_per_turn: int = Field(default=8, ge=1, le=20)
    viewport_radius_lines: int = Field(default=45, ge=10, le=200)
    max_file_chars: int = Field(default=40_000, ge=2_000, le=120_000)
    hitl_policy: EditorHitlPolicy = "all_writes"

