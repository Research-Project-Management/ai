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

    @property
    def is_production(self) -> bool:
        return self.app_env == AppEnvironment.PRODUCTION

    @model_validator(mode="after")
    def _validate_production_security(self) -> Self:
        if self.is_production and not self.internal_api_key.get_secret_value().strip():
            raise ValueError("INTERNAL_API_KEY must be set when APP_ENV=production")
        return self
