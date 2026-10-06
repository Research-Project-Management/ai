"""Root configuration aggregate and settings loader."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai.shared.config.database import DatabaseConfig
from ai.shared.config.editor import EditorConfig
from ai.shared.config.embedding import EmbeddingConfig
from ai.shared.config.library import LibraryConfig
from ai.shared.config.llm import LLMConfig
from ai.shared.config.project import ProjectConfig
from ai.shared.config.rag import IngestionConfig, RerankConfig
from ai.shared.config.rate_limit import RateLimitConfig
from ai.shared.config.server import ServerConfig
from ai.shared.config.web_search import WebSearchConfig


class Settings(BaseModel):
    """Root immutable configuration aggregate validated at startup."""

    model_config = ConfigDict(frozen=True)

    server: ServerConfig = Field(default_factory=ServerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    library: LibraryConfig = Field(default_factory=LibraryConfig)
    editor: EditorConfig = Field(default_factory=EditorConfig)
    project: ProjectConfig = Field(default_factory=ProjectConfig)

    @model_validator(mode="after")
    def _validate_production_readiness(self) -> Self:
        if self.server.is_production:
            if not self.llm.api_key.get_secret_value().strip():
                raise ValueError("LLM_API_KEY must be set when APP_ENV=production")
            if not self.database.postgres.is_configured:
                raise ValueError("DATABASE_URL must be set when APP_ENV=production")
            if self.rate_limit.enabled and not self.database.redis.is_configured:
                raise ValueError(
                    "REDIS_URL must be set when APP_ENV=production and RATE_LIMIT_ENABLED=true"
                )
        return self


def load_settings() -> Settings:
    """Pure factory to load and validate settings at the Composition Root."""
    return Settings()


__all__ = ["Settings", "load_settings"]
