"""Immutable value objects for curated, deduplicated, and sanitized web evidence."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ai.agents.web_search.fetch.contracts import BlockedReason
from ai.orchestrator.contracts import CitationItem


class RankedWebEvidence(BaseModel):
    """Ranked and sanitized external web/scholarly passage ready for synthesis."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)
    snippet: str = ""
    content: str = ""
    provider: str = ""
    category: Literal["web", "scholar"] = "web"
    doi: str | None = None
    arxiv_id: str | None = None
    authors: tuple[str, ...] = ()
    published_year: int | None = None
    open_access_pdf_url: str | None = None
    score: float = Field(default=0.0, ge=0.0)

    def to_citation(self) -> CitationItem:
        """Project to an orchestrator `CitationItem` with verified external provenance."""
        preview = (self.snippet or self.content)[:280].strip()
        return CitationItem(
            source_id=self.source_id,
            title=self.title,
            url=self.url,
            snippet=preview,
        )


class BlockedSourceReport(BaseModel):
    """Structured notice for external sources blocked by WAF, paywall, or security guards."""

    model_config = ConfigDict(frozen=True)

    url: str = Field(..., min_length=1)
    reason: BlockedReason
    doi: str | None = None
    arxiv_id: str | None = None


class CuratedEvidenceBatch(BaseModel):
    """Reverse-repacked and descending-ranked external evidence bundle."""

    model_config = ConfigDict(frozen=True)

    repacked: tuple[RankedWebEvidence, ...] = ()
    ranked_desc: tuple[RankedWebEvidence, ...] = ()
    blocked_sources: tuple[BlockedSourceReport, ...] = ()
