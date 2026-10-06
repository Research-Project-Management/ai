"""Editor agent configuration settings."""

from typing import Literal

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from ai.shared.config.agent import BackendAgentConfig


class EditorConfig(BackendAgentConfig):
    """Flux LaTeX Editor Backend Gateway and agent execution bounds (`EDITOR_*`)."""

    model_config = SettingsConfigDict(env_prefix="EDITOR_")

    max_steps: int = Field(default=6, ge=1, le=12)
    max_patches_per_turn: int = Field(default=8, ge=1, le=20)
    viewport_radius_lines: int = Field(default=45, ge=10, le=200)
    max_file_chars: int = Field(default=40_000, ge=2_000, le=120_000)
    hitl_policy: Literal["all_writes", "readonly"] = "all_writes"
