"""Rate limiting configuration scoped per Flux space."""

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from ai.shared.config.base import EnvConfigBase


class RateLimitConfig(EnvConfigBase):
    """Sliding-window rate limiter configuration (`RATE_LIMIT_*`)."""

    model_config = SettingsConfigDict(env_prefix="RATE_LIMIT_")

    enabled: bool = True
    requests_per_window: int = Field(default=30, ge=1, le=100_000)
    window_seconds: int = Field(default=60, ge=1, le=86_400)
    fail_open: bool = True
