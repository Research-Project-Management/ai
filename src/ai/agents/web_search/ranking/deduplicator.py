"""URL canonicalization and cross-provider DOI / arXiv / URL deduplication."""

import re
from collections.abc import Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from ai.agents.web_search.fetch.extractor import extract_academic_identifiers
from ai.agents.web_search.search.contracts import SearchHit

_TRACKING_PARAM_PREFIXES = ("utm_", "mtm_", "pk_")
_TRACKING_PARAMS = frozenset(
    {"fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid", "ref", "source"}
)


def canonicalize_url(raw_url: str) -> str:
    """Normalize scheme, host, tracking parameters, and trailing slash."""
    cleaned = raw_url.strip()
    if not cleaned:
        return ""
    parsed = urlparse(cleaned)
    scheme = (parsed.scheme or "https").lower()
    netloc = (parsed.netloc or "").lower()
    if (scheme == "http" and netloc.endswith(":80")) or (
        scheme == "https" and netloc.endswith(":443")
    ):
        netloc = netloc.rsplit(":", 1)[0]

    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    kept_params = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=False)
        if k.lower() not in _TRACKING_PARAMS and not k.lower().startswith(_TRACKING_PARAM_PREFIXES)
    ]
    kept_params.sort()
    query = urlencode(kept_params)
    return urlunparse((scheme, netloc, path, "", query, ""))


def compute_canonical_id(
    *,
    url: str,
    doi: str | None = None,
    arxiv_id: str | None = None,
) -> str:
    """Compute a unified identity key so Web and Scholar hits collapse cleanly."""
    clean_doi = (doi or "").strip().lower()
    if clean_doi.startswith(("http://", "https://")):
        clean_doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", clean_doi)
    clean_arxiv = (arxiv_id or "").strip().lower()

    if not clean_doi or not clean_arxiv:
        derived = extract_academic_identifiers(url)
        clean_doi = clean_doi or (derived.doi or "")
        clean_arxiv = clean_arxiv or (derived.arxiv_id or "")

    if clean_doi:
        return f"doi:{clean_doi}"
    if clean_arxiv:
        return f"arxiv:{re.sub(r'v\d+$', '', clean_arxiv)}"
    return canonicalize_url(url) or url.strip()


def _merge_hits(primary: SearchHit, incoming: SearchHit) -> SearchHit:
    better_content = (
        incoming.content if len(incoming.content) > len(primary.content) else primary.content
    )
    better_snippet = (
        incoming.snippet if len(incoming.snippet) > len(primary.snippet) else primary.snippet
    )
    better_authors = (
        incoming.authors if len(incoming.authors) > len(primary.authors) else primary.authors
    )
    better_category = (
        "scholar" if (primary.category == "scholar" or incoming.category == "scholar") else "web"
    )
    better_url = (
        incoming.url
        if primary.category == "web" and incoming.category == "scholar"
        else primary.url
    )
    better_title = incoming.title if len(incoming.title) > len(primary.title) else primary.title
    return SearchHit(
        provider=f"{primary.provider},{incoming.provider}"
        if incoming.provider not in primary.provider.split(",")
        else primary.provider,
        category=better_category,
        title=better_title,
        url=better_url,
        snippet=better_snippet,
        content=better_content,
        doi=primary.doi or incoming.doi,
        arxiv_id=primary.arxiv_id or incoming.arxiv_id,
        pmid=primary.pmid or incoming.pmid,
        authors=better_authors,
        published_year=primary.published_year or incoming.published_year,
        open_access_pdf_url=primary.open_access_pdf_url or incoming.open_access_pdf_url,
        score=max(primary.score, incoming.score),
    )


def deduplicate_hits(hits: Sequence[SearchHit]) -> list[SearchHit]:
    """Collapse duplicates across providers while merging scholarly & web metadata."""
    merged: dict[str, SearchHit] = {}
    order: list[str] = []
    for hit in hits:
        key = compute_canonical_id(url=hit.url, doi=hit.doi, arxiv_id=hit.arxiv_id)
        if key not in merged:
            merged[key] = hit
            order.append(key)
        else:
            merged[key] = _merge_hits(merged[key], hit)
    return [merged[key] for key in order]
