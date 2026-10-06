"""Base configuration class for immutable, environment-backed settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvConfigBase(BaseSettings):
    """Canonical base class shared by all domain config models."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        case_sensitive=False,
    )


def validate_http_url(
    value: str,
    *,
    allow_empty: bool = True,
    strip_trailing_slash: bool = False,
    label: str = "URL setting",
) -> str:
    """Validate and normalize an HTTP/HTTPS URL configuration value."""
    stripped = value.strip()
    if strip_trailing_slash:
        stripped = stripped.rstrip("/")
    if not stripped:
        if not allow_empty:
            raise ValueError(f"{label} must not be empty")
        return ""
    if not stripped.startswith(("http://", "https://")):
        raise ValueError(f"{label} must start with http:// or https://")
    return stripped
