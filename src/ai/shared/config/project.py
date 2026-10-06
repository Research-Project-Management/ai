"""Project agent configuration settings."""

from typing import Literal

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from ai.shared.config.agent import BackendAgentConfig


class ProjectConfig(BackendAgentConfig):
    """Flux Project Management Backend Gateway and agent execution bounds (`PROJECT_*`)."""

    model_config = SettingsConfigDict(env_prefix="PROJECT_")

    max_steps: int = Field(default=6, ge=1, le=12)
    hitl_policy: Literal["destructive", "all_writes", "readonly"] = "destructive"
