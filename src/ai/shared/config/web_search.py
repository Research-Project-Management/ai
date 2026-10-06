"""Web search, scholarly mirror, and safe URL fetch configuration."""

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import SettingsConfigDict

from ai.shared.config.base import EnvConfigBase, validate_http_url


class WebSearchConfig(EnvConfigBase):
    """External search providers, open scholarly mirrors, and fetch bounds (`WEB_SEARCH_*`)."""

    model_config = SettingsConfigDict(env_prefix="WEB_SEARCH_")

    searxng_url: str = ""
    tavily_api_key: SecretStr = Field(default=SecretStr(""))
    tavily_base_url: str = Field(default="https://api.tavily.com")
    serper_api_key: SecretStr = Field(default=SecretStr(""))
    serper_base_url: str = Field(default="https://google.serper.dev")
    openalex_api_key: SecretStr = Field(default=SecretStr(""))
    openalex_base_url: str = Field(default="https://api.openalex.org")
    arxiv_base_url: str = Field(default="https://export.arxiv.org/api/query")
    crossref_base_url: str = Field(default="https://api.crossref.org")
    europepmc_base_url: str = Field(default="https://www.ebi.ac.uk/europepmc/webservices/rest")
    user_agent: str = Field(default="FluxResearchAI/0.1 (+https://flux.local)", min_length=1)
    timeout_seconds: float = Field(default=10.0, gt=0.0, le=60.0)
    max_results_per_provider: int = Field(default=6, ge=1, le=25)
    top_k_evidence: int = Field(default=6, ge=1, le=20)
    max_deep_fetch_urls: int = Field(default=3, ge=0, le=10)
    max_fetch_bytes: int = Field(default=2 * 1024 * 1024, ge=4096, le=20 * 1024 * 1024)
    cache_ttl_seconds: int = Field(default=1800, ge=0, le=86400)
    min_relevance_score: float = Field(default=0.05, ge=0.0, le=1.0)

    @property
    def has_searxng(self) -> bool:
        return bool(self.searxng_url.strip())

    @property
    def has_tavily(self) -> bool:
        return bool(self.tavily_api_key.get_secret_value().strip())

    @property
    def has_serper(self) -> bool:
        return bool(self.serper_api_key.get_secret_value().strip())

    @field_validator(
        "searxng_url",
        "tavily_base_url",
        "serper_base_url",
        "openalex_base_url",
        "arxiv_base_url",
        "crossref_base_url",
        "europepmc_base_url",
    )
    @classmethod
    def _validate_http_url(cls, value: str) -> str:
        return validate_http_url(value, strip_trailing_slash=True, label="URL setting")
