"""Backend-backed agent configuration settings (`EDITOR_*`, `LIBRARY_*`, `PROJECT_*`)."""

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import SettingsConfigDict

from ai.shared.config.base import EnvConfigBase, validate_http_url


class BackendAgentConfig(EnvConfigBase):
    """Backend gateway settings common to the editor, library, and project agents."""

    enabled: bool = True
    backend_url: str = Field(default="http://127.0.0.1:18300")
    timeout_seconds: float = Field(default=15.0, gt=0.0, le=120.0)

    @field_validator("backend_url")
    @classmethod
    def _validate_backend_url(cls, value: str) -> str:
        prefix = str(cls.model_config.get("env_prefix", ""))
        return validate_http_url(
            value,
            strip_trailing_slash=True,
            label=f"{prefix}BACKEND_URL",
        )


class EditorConfig(BackendAgentConfig):
    """Flux LaTeX Editor Backend Gateway and agent execution bounds (`EDITOR_*`)."""

    model_config = SettingsConfigDict(env_prefix="EDITOR_")

    max_steps: int = Field(default=6, ge=1, le=12)
    max_patches_per_turn: int = Field(default=8, ge=1, le=20)
    viewport_radius_lines: int = Field(default=45, ge=10, le=200)
    max_file_chars: int = Field(default=40_000, ge=2_000, le=120_000)
    hitl_policy: Literal["all_writes", "readonly"] = "all_writes"


class LibraryConfig(BackendAgentConfig):
    """Flux Library Backend Gateway and agent execution bounds (`LIBRARY_*`)."""

    model_config = SettingsConfigDict(env_prefix="LIBRARY_")

    max_steps: int = Field(default=5, ge=1, le=12)
    max_items_per_call: int = Field(default=20, ge=1, le=50)
    hitl_policy: Literal["destructive", "all_writes", "readonly"] = "destructive"


class ProjectConfig(BackendAgentConfig):
    """Flux Project Management Backend Gateway and agent execution bounds (`PROJECT_*`)."""

    model_config = SettingsConfigDict(env_prefix="PROJECT_")

    max_steps: int = Field(default=6, ge=1, le=12)
    hitl_policy: Literal["destructive", "all_writes", "readonly"] = "destructive"
