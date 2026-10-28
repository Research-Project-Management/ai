"""Fetch stage serving content the caller already holds."""

import re

from ai.agents.web_search.fetch.contracts import (
    AcademicIdentifiers,
    ClientSourcePayload,
    FetchRequest,
    FetchResult,
)
from ai.agents.web_search.ranking.deduplicator import canonicalize_url


def _matches_client_source(request: FetchRequest, source: ClientSourcePayload) -> bool:
    if canonicalize_url(source.url) == canonicalize_url(request.url):
        return True
    if (
        request.identifiers.doi
        and source.doi
        and request.identifiers.doi.lower() == source.doi.strip().lower()
    ):
        return True
    if request.identifiers.arxiv_id and source.arxiv_id:
        lhs = re.sub(r"v\d+$", "", request.identifiers.arxiv_id.lower())
        rhs = re.sub(r"v\d+$", "", source.arxiv_id.strip().lower())
        if lhs == rhs:
            return True
    return False


class ClientSuppliedFetchStage:
    """Stage 1: Resolve immediately from User-IP / browser-enriched payloads (0ms)."""

    @property
    def name(self) -> str:
        return "client_supplied"

    async def try_fetch(self, request: FetchRequest) -> FetchResult | None:
        for source in request.client_sources:
            if not _matches_client_source(request, source):
                continue
            body = (source.content or source.snippet).strip()
            if not body:
                continue
            merged_ids = request.identifiers.merge(
                AcademicIdentifiers(
                    doi=source.doi.lower() if source.doi else None,
                    arxiv_id=source.arxiv_id,
                )
            )
            return FetchResult(
                url=request.url,
                final_url=source.url,
                title=source.title.strip() or request.url,
                markdown=body,
                stage_resolved=self.name,
                is_blocked=False,
                identifiers=merged_ids,
                authors=source.authors,
                published_year=source.published_year,
            )
        return None
