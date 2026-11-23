"""RAG ingestion and reranking configuration."""

from typing import Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import SettingsConfigDict

from ai.shared.config.base import EnvConfigBase, validate_http_url


class IngestionConfig(EnvConfigBase):
    """Document parsing and structure-aware chunking settings (`INGESTION_*`)."""

    model_config = SettingsConfigDict(env_prefix="INGESTION_")

    chunk_size_tokens: int = Field(default=384, ge=64, le=2048)
    chunk_overlap_tokens: int = Field(default=48, ge=0, le=512)
    max_file_bytes: int = Field(default=50 * 1024 * 1024, ge=1024)
    max_concurrency: int = Field(default=2, ge=1, le=16)
    max_queue_size: int = Field(default=50, ge=1, le=1000)
    job_ttl_seconds: int = Field(default=86400, ge=60, le=604800)

    @model_validator(mode="after")
    def _validate_overlap_window(self) -> Self:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                "INGESTION_CHUNK_OVERLAP_TOKENS must be smaller than chunk_size_tokens"
            )
        return self


class RerankConfig(EnvConfigBase):
    """Cross-encoder reranking and reverse-repacking settings (`RERANK_*`)."""

    model_config = SettingsConfigDict(env_prefix="RERANK_")

    base_url: str = ""
    api_key: SecretStr = Field(default=SecretStr(""))
    model: str = Field(default="bge-reranker-v2-m3", min_length=1)
    timeout_seconds: float = Field(default=8.0, gt=0.0, le=60.0)
    candidate_pool_size: int = Field(default=24, ge=5, le=100)
    top_k: int = Field(default=6, ge=1, le=30)
    min_score: float = Field(default=0.05, ge=0.0, le=1.0)

    @property
    def is_remote_configured(self) -> bool:
        return bool(self.base_url.strip())

    @field_validator("base_url")
    @classmethod
    def _validate_http_url(cls, value: str) -> str:
        return validate_http_url(value, label="RERANK_BASE_URL")
