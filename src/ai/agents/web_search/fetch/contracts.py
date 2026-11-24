"""Immutable contracts and Escalation Stage interface for safe URL & paper fetching."""

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

BlockedReason = Literal[
    "ssrf_blocked",
    "domain_blocked",
    "waf_challenge",
    "paywall_or_forbidden",
    "rate_limited",
    "unsupported_content",
    "network_error",
]

TERMINAL_SECURITY_BLOCKS: frozenset[BlockedReason] = frozenset({"ssrf_blocked", "domain_blocked"})


class AcademicIdentifiers(BaseModel):
    """Normalized scholarly identifiers extracted from URLs, prompts, or APIs."""

    model_config = ConfigDict(frozen=True)

    doi: str | None = None
    arxiv_id: str | None = None
    pmid: str | None = None
    pmcid: str | None = None

    @property
    def has_any(self) -> bool:
        return any((self.doi, self.arxiv_id, self.pmid, self.pmcid))

    def merge(self, other: "AcademicIdentifiers") -> "AcademicIdentifiers":
        return AcademicIdentifiers(
            doi=self.doi or other.doi,
            arxiv_id=self.arxiv_id or other.arxiv_id,
            pmid=self.pmid or other.pmid,
            pmcid=self.pmcid or other.pmcid,
        )


class ClientSourcePayload(BaseModel):
    """Pre-fetched source data supplied directly by the caller (e.g., via User-IP browser)."""

    model_config = ConfigDict(frozen=True)

    url: str = Field(..., min_length=1)
    title: str = ""
    content: str = ""
    snippet: str = ""
    doi: str | None = None
    arxiv_id: str | None = None
    authors: tuple[str, ...] = ()
    published_year: int | None = None


class FetchRequest(BaseModel):
    """Normalized input for executing one URL through `UrlFetchLadder`."""

    model_config = ConfigDict(frozen=True)

    url: str = Field(..., min_length=1)
    identifiers: AcademicIdentifiers = Field(default_factory=AcademicIdentifiers)
    client_sources: tuple[ClientSourcePayload, ...] = ()
    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()


class FetchResult(BaseModel):
    """Outcome from resolving a URL across the Escalation Ladder."""

    model_config = ConfigDict(frozen=True)

    url: str
    final_url: str = ""
    title: str = ""
    markdown: str = ""
    stage_resolved: str = ""
    is_blocked: bool = False
    blocked_reason: BlockedReason | None = None
    identifiers: AcademicIdentifiers = Field(default_factory=AcademicIdentifiers)
    authors: tuple[str, ...] = ()
    published_year: int | None = None
    open_access_pdf_url: str | None = None


class UrlFetchStage(Protocol):
    """Single stage in the ordered URL fetch escalation ladder."""

    @property
    def name(self) -> str: ...

    async def try_fetch(self, request: FetchRequest) -> FetchResult | None: ...
