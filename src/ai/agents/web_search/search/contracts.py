"""Immutable contracts and domain-scoping value rules for web/scholar search."""

import re
from collections.abc import Sequence
from typing import Any, Literal, Protocol, Self
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ProviderCategory = Literal["web", "scholar"]
SearchCategoryFilter = Literal["web", "scholar", "hybrid"]

_DOMAIN_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


def normalize_domain_entry(raw: str) -> str:
    """Normalize a user-supplied domain or URL string into a bare lowercase hostname."""
    cleaned = raw.strip().lower()
    if not cleaned:
        raise ValueError("Domain entry cannot be empty")
    if "://" in cleaned:
        parsed = urlparse(cleaned)
        host = (parsed.hostname or "").strip(".")
    else:
        host_part = cleaned.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
        if ":" in host_part:
            host_part = host_part.split(":", 1)[0]
        host = host_part.lstrip("*.")

    if not host or "." not in host:
        raise ValueError(f"Invalid domain entry: '{raw}'")
    labels = host.split(".")
    if any(not _DOMAIN_LABEL_RE.match(label) for label in labels):
        raise ValueError(f"Invalid characters in domain entry: '{raw}'")
    return host


def _normalize_domain_tuple(values: Sequence[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        domain = normalize_domain_entry(text)
        if domain not in seen:
            seen.add(domain)
            normalized.append(domain)
    return tuple(normalized)


class SearchQuery(BaseModel):
    """Normalized query submitted to `SearchProvider` implementations."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., min_length=1)
    category: SearchCategoryFilter = "hybrid"
    max_results: int = Field(default=6, ge=1, le=25)
    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Search query cannot be blank")
        return stripped

    @field_validator("allowed_domains", "blocked_domains", mode="before")
    @classmethod
    def _validate_domains(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return _normalize_domain_tuple([value])
        if isinstance(value, Sequence):
            return _normalize_domain_tuple(value)
        raise ValueError("Domain filter must be a sequence of domain strings")

    @model_validator(mode="after")
    def _validate_domain_exclusivity(self) -> Self:
        if self.allowed_domains and self.blocked_domains:
            raise ValueError(
                "Cannot specify both allowed_domains and blocked_domains simultaneously"
            )
        return self


class SearchHit(BaseModel):
    """Normalized candidate hit returned by any web or scholarly provider."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(..., min_length=1)
    category: ProviderCategory
    title: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)
    snippet: str = ""
    content: str = ""
    doi: str | None = None
    arxiv_id: str | None = None
    pmid: str | None = None
    authors: tuple[str, ...] = ()
    published_year: int | None = None
    open_access_pdf_url: str | None = None
    score: float = Field(default=0.0, ge=0.0)


class SearchProvider(Protocol):
    """Pluggable search adapter interface (Strategy pattern)."""

    @property
    def name(self) -> str: ...

    @property
    def category(self) -> ProviderCategory: ...

    async def search(self, query: SearchQuery) -> tuple[SearchHit, ...]: ...


class SearchCache(Protocol):
    """Fail-open L2 cache interface shared across `search/` and `fetch/`."""

    async def get_hits(self, key: str) -> tuple[SearchHit, ...] | None: ...

    async def set_hits(
        self, key: str, hits: tuple[SearchHit, ...], *, ttl_seconds: int
    ) -> None: ...

    async def get_json(self, key: str) -> dict[str, Any] | None: ...

    async def set_json(self, key: str, payload: dict[str, Any], *, ttl_seconds: int) -> None: ...
