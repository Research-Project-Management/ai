"""Shared base for agents that call the Flux Backend (`<PREFIX>_BACKEND_URL`, timeout, enable flag)."""

from pydantic import Field, field_validator

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
