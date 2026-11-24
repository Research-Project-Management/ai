"""Embedding model, fallback provider, and Qdrant collection configuration."""

from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import SettingsConfigDict

from ai.shared.config.base import EnvConfigBase, validate_http_url
from ai.shared.config.llm import LLMConfig


class EmbeddingConfig(EnvConfigBase):
    """OpenAI-compatible embedding, optional fallback, and Qdrant target settings (`EMBEDDING_*`)."""

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")

    provider: Literal["auto", "openai", "local_hash"] = "auto"
    base_url: str = ""
    api_key: SecretStr = Field(default=SecretStr(""))
    model: str = Field(default="bge-m3", min_length=1)
    dimensions: int = Field(default=1024, ge=8, le=8192)
    timeout_seconds: float = Field(default=60.0, gt=0.0, le=300.0)
    max_retries: int = Field(default=2, ge=0, le=10)
    batch_size: int = Field(default=16, ge=1, le=256)
    max_input_chars: int = Field(default=0, ge=0, le=100_000)
    send_dimensions: bool | None = None

    fallback_base_url: str = ""
    fallback_api_key: SecretStr = Field(default=SecretStr(""))
    fallback_model: str = ""
    fallback_max_retries: int = Field(default=1, ge=0, le=10)

    collection_name: str = Field(default="kb_bge_m3", min_length=1)
    collection_alias: str = ""

    @field_validator("base_url", "fallback_base_url")
    @classmethod
    def _validate_http_url(cls, value: str) -> str:
        return validate_http_url(value, label="EMBEDDING base URL")

    @property
    def has_fallback(self) -> bool:
        return bool(
            self.fallback_base_url.strip() or self.fallback_api_key.get_secret_value().strip()
        )

    @property
    def effective_fallback_model(self) -> str:
        return self.fallback_model.strip() or self.model

    def should_send_dimensions(self, model_name: str | None = None) -> bool:
        if self.send_dimensions is not None:
            return self.send_dimensions
        target_model = (model_name or self.model).strip().lower()
        return target_model.startswith("text-embedding-3")

    def effective_base_url(self, llm: LLMConfig) -> str:
        """Fallback to the primary LLM relay URL when EMBEDDING_BASE_URL is unset."""
        return self.base_url.strip() or llm.base_url.strip()

    def effective_api_key(self, llm: LLMConfig) -> str:
        """Use EMBEDDING_API_KEY, avoiding leaking LLM_API_KEY to an explicit local endpoint."""
        explicit_key = self.api_key.get_secret_value().strip()
        if explicit_key:
            return explicit_key
        if self.base_url.strip():
            return ""
        return llm.api_key.get_secret_value().strip()
