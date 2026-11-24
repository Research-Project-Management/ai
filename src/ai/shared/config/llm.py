"""LLM provider and fallback configuration."""

from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import SettingsConfigDict

from ai.shared.config.base import EnvConfigBase, validate_http_url


class LLMConfig(EnvConfigBase):
    """OpenAI-compatible primary LLM and fallback provider configuration."""

    model_config = SettingsConfigDict(env_prefix="LLM_")

    base_url: str = ""
    api_key: SecretStr = Field(default=SecretStr(""))
    extra_api_keys: SecretStr = Field(default=SecretStr(""))
    model: str = Field(default="tozpie-1.0-flash", min_length=1)
    router_model: str = ""
    router_base_url: str = ""
    router_api_key: SecretStr = Field(default=SecretStr(""))
    agent_model: str = ""
    agent_base_url: str = ""
    agent_api_key: SecretStr = Field(default=SecretStr(""))
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=45.0, gt=0.0, le=300.0)
    max_retries: int = Field(default=2, ge=0, le=10)
    max_tokens: int = Field(default=0, ge=0, le=32768)

    fallback_base_url: str = ""
    fallback_api_key: SecretStr = Field(default=SecretStr(""))
    fallback_model: str = ""
    fallback_max_retries: int = Field(default=1, ge=0, le=10)

    @property
    def effective_router_model(self) -> str:
        return self.router_model.strip() or self.model

    @property
    def effective_agent_model(self) -> str:
        return self.agent_model.strip() or self.model

    def effective_base_url(self, role: Literal["default", "router", "agent"] = "default") -> str:
        """Return role-specific base URL if set, otherwise fall back to primary `base_url`."""
        role_url = {
            "router": self.router_base_url,
            "agent": self.agent_base_url,
        }.get(role, "")
        return role_url.strip() or self.base_url.strip()

    def role_api_key_pool(
        self,
        role: Literal["default", "router", "agent"] = "default",
    ) -> tuple[SecretStr, ...]:
        """Return role-specific key when configured, otherwise fall back to `api_key_pool`."""
        role_key = {
            "router": self.router_api_key,
            "agent": self.agent_api_key,
        }.get(role)
        if role_key is not None:
            raw = role_key.get_secret_value().strip()
            if raw:
                return (SecretStr(raw),)
        return self.api_key_pool

    @property
    def parsed_extra_api_keys(self) -> tuple[SecretStr, ...]:
        raw = self.extra_api_keys.get_secret_value()
        if not raw.strip():
            return ()
        return tuple(SecretStr(part.strip()) for part in raw.split(",") if part.strip())

    @property
    def api_key_pool(self) -> tuple[SecretStr, ...]:
        """Return deduplicated primary + extra API keys in configured order."""
        seen: set[str] = set()
        pool: list[SecretStr] = []
        primary_raw = self.api_key.get_secret_value().strip() or "not-set"
        seen.add(primary_raw)
        pool.append(SecretStr(primary_raw))
        for extra in self.parsed_extra_api_keys:
            val = extra.get_secret_value().strip()
            if val and val not in seen:
                seen.add(val)
                pool.append(SecretStr(val))
        return tuple(pool)

    @property
    def has_fallback(self) -> bool:
        return bool(self.fallback_api_key.get_secret_value().strip())

    @field_validator("base_url", "fallback_base_url", "router_base_url", "agent_base_url")
    @classmethod
    def _validate_http_url(cls, value: str) -> str:
        return validate_http_url(value, label="LLM base URL")
