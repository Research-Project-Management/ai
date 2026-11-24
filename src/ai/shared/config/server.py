"""Server and API gateway configuration."""

from enum import StrEnum
from typing import Self

from pydantic import Field, SecretStr, model_validator

from ai.shared.config.base import EnvConfigBase


class AppEnvironment(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class ServerConfig(EnvConfigBase):
    """Runtime server, environment mode, and internal gateway auth settings."""

    app_env: AppEnvironment = AppEnvironment.DEVELOPMENT
    host: str = Field(default="0.0.0.0", min_length=1)
    port: int = Field(default=18420, ge=1024, le=65535)
    internal_api_key: SecretStr = Field(default=SecretStr(""))
    http_max_connections: int = Field(default=200, ge=10, le=2000)
    http_max_keepalive_connections: int = Field(default=50, ge=5, le=500)
    http_keepalive_expiry_seconds: float = Field(default=30.0, gt=0.0, le=300.0)
    http_timeout_seconds: float = Field(default=15.0, gt=0.0, le=120.0)

    @property
    def is_production(self) -> bool:
        return self.app_env == AppEnvironment.PRODUCTION

    @model_validator(mode="after")
    def _validate_production_security(self) -> Self:
        if self.is_production and not self.internal_api_key.get_secret_value().strip():
            raise ValueError("INTERNAL_API_KEY must be set when APP_ENV=production")
        return self
