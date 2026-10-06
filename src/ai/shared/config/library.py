"""Library agent configuration settings."""

from typing import Literal

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from ai.shared.config.agent import BackendAgentConfig


class LibraryConfig(BackendAgentConfig):
    """Flux Library Backend Gateway and agent execution bounds (`LIBRARY_*`)."""

    model_config = SettingsConfigDict(env_prefix="LIBRARY_")

    max_steps: int = Field(default=5, ge=1, le=12)
    max_items_per_call: int = Field(default=20, ge=1, le=50)
    hitl_policy: Literal["destructive", "all_writes", "readonly"] = "destructive"
