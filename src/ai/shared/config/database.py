"""Database and persistence configuration segregated per storage engine."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

from ai.shared.config.base import EnvConfigBase, validate_http_url


class PostgresConfig(EnvConfigBase):
    """PostgreSQL configuration for LangGraph checkpointer and persistence."""

    database_url: SecretStr = Field(default=SecretStr(""))
    pg_pool_min_size: int = Field(default=2, ge=1, le=50)
    pg_pool_max_size: int = Field(default=10, ge=1, le=100)
    pg_pool_timeout_seconds: float = Field(default=10.0, gt=0.0)
    pg_setup_on_startup: bool = Field(default=True)

    @property
    def is_configured(self) -> bool:
        return bool(self.database_url.get_secret_value().strip())

    @field_validator("database_url")
    @classmethod
    def _validate_postgres_url(cls, value: SecretStr) -> SecretStr:
        raw = value.get_secret_value().strip()
        if raw and not raw.startswith(("postgresql://", "postgres://")):
            raise ValueError("DATABASE_URL must start with postgresql:// or postgres://")
        return SecretStr(raw)

    @model_validator(mode="after")
    def _validate_pool_bounds(self) -> Self:
        if self.pg_pool_max_size < self.pg_pool_min_size:
            raise ValueError("pg_pool_max_size must be greater than or equal to pg_pool_min_size")
        return self


class RedisConfig(EnvConfigBase):
    """Redis configuration for caching and rate limiting."""

    redis_url: SecretStr = Field(default=SecretStr("redis://localhost:31942"))
    redis_socket_connect_timeout: float = Field(default=1.5, gt=0.0, le=30.0)
    redis_socket_timeout: float = Field(default=2.0, gt=0.0, le=30.0)

    @property
    def is_configured(self) -> bool:
        return bool(self.redis_url.get_secret_value().strip())

    @field_validator("redis_url")
    @classmethod
    def _validate_redis_url(cls, value: SecretStr) -> SecretStr:
        raw = value.get_secret_value().strip()
        if raw and not raw.startswith(("redis://", "rediss://")):
            raise ValueError("REDIS_URL must start with redis:// or rediss://")
        return SecretStr(raw)


class QdrantConfig(EnvConfigBase):
    """Qdrant vector store configuration."""

    qdrant_url: str = Field(default="http://localhost:27519")
    qdrant_api_key: SecretStr = Field(default=SecretStr(""))
    qdrant_timeout_seconds: int = Field(default=10, ge=1, le=120)

    @property
    def is_configured(self) -> bool:
        return bool(self.qdrant_url.strip())

    @field_validator("qdrant_url")
    @classmethod
    def _validate_qdrant_url(cls, value: str) -> str:
        return validate_http_url(value, label="QDRANT_URL")


class DatabaseConfig(BaseModel):
    """Aggregate persistence settings grouped by engine."""

    model_config = ConfigDict(frozen=True)

    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
