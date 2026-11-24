"""Configuration for the `editor_agent` workflow."""

from __future__ import annotations

from pydantic import Field

from ai.shared.workflow.plan import ToolAgentConfig, ToolHitlPolicy


class EditorAgentConfig(ToolAgentConfig):
    """Immutable bounded execution and safety configuration for `editor_agent`."""

    max_patches_per_turn: int = Field(default=8, ge=1, le=20)
    viewport_radius_lines: int = Field(default=45, ge=10, le=200)
    max_file_chars: int = Field(default=40_000, ge=2_000, le=120_000)
    hitl_policy: ToolHitlPolicy = "all_writes"
