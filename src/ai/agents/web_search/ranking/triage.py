"""Scoring, selective deep-fetch triage, Anti-IPI sanitization & Reverse Repacking."""

import re
from collections.abc import Sequence
from itertools import pairwise
from urllib.parse import urlparse

from ai.agents.web_search.fetch.contracts import FetchResult
from ai.agents.web_search.fetch.extractor import extract_academic_identifiers
from ai.agents.web_search.ranking.contracts import (
    BlockedSourceReport,
    CuratedEvidenceBatch,
    RankedWebEvidence,
)
from ai.agents.web_search.ranking.deduplicator import (
    canonicalize_url,
    compute_canonical_id,
    deduplicate_hits,
)
from ai.agents.web_search.search.contracts import SearchHit

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}|[\u00c0-\u1ef9]{2,}", re.IGNORECASE)
_INJECTION_TAG_RE = re.compile(r"<(/?)untrusted_web_content\b", re.IGNORECASE)
_DIRECTIVE_LINE_RE = re.compile(
    r"^\s*(?:ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions|"
    r"disregard\s+(?:all\s+)?(?:previous|prior)\s+instructions|"
    r"you\s+are\s+now\b|system\s+prompt\s*:|assistant\s+instruction\s*:).*$",
    re.IGNORECASE | re.MULTILINE,
)

_HIGH_TRUST_DOMAINS = (
    "arxiv.org",
    "openalex.org",
    "doi.org",
    "acm.org",
    "ieee.org",
    "nature.com",
    "springer.com",
    "sciencedirect.com",
    "ncbi.nlm.nih.gov",
    "github.com",
    "huggingface.co",
    "pytorch.org",
    "python.org",
    "docs.",
)


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def sanitize_untrusted_web_text(
    text: str,
    *,
    source_url: str,
    max_chars: int = 4000,
) -> str:
    """Wrap external text in `<untrusted_web_content>` and neutralize IPI directives."""
    escaped_tags = _INJECTION_TAG_RE.sub(r"&lt;\1untrusted_web_content", text.strip())
    neutralized = _DIRECTIVE_LINE_RE.sub("[filtered-untrusted-directive]", escaped_tags)
    if len(neutralized) > max_chars:
        neutralized = neutralized[:max_chars].rstrip() + "..."
    safe_url = source_url.replace('"', "%22")
    return f'<untrusted_web_content url="{safe_url}">\n{neutralized}\n</untrusted_web_content>'


def _authority_bonus(hit: SearchHit) -> float:
    bonus = 0.0
    if hit.category == "scholar" or hit.doi or hit.arxiv_id:
        bonus += 0.12
    host = (urlparse(hit.url).hostname or "").lower()
    if host.endswith((".edu", ".gov")) or ".ac." in host or any(d in host for d in _HIGH_TRUST_DOMAINS):
        bonus += 0.08
    if hit.published_year is not None and hit.published_year >= 2022:
        bonus += 0.05
    return bonus


def score_search_hit(query: str, hit: SearchHit) -> float:
    """Deterministically score a hit in `[0.0, 1.0]` combining lexical match & provenance."""
    q_tokens = _tokenize(query)
    haystack_tokens = _tokenize(f"{hit.title} {hit.snippet} {hit.content[:1500]}")
    if not haystack_tokens:
        return min(1.0, max(0.0, hit.score * 0.25))

    haystack_set = set(haystack_tokens)
    if q_tokens:
        unigram_recall = sum(1 for tok in q_tokens if tok in haystack_set) / len(q_tokens)
        q_bigrams = list(pairwise(q_tokens))
        haystack_bigrams = set(pairwise(haystack_tokens))
        bigram_recall = (
            sum(1 for bg in q_bigrams if bg in haystack_bigrams) / len(q_bigrams)
            if q_bigrams
            else unigram_recall
        )
        lexical = 0.7 * unigram_recall + 0.3 * bigram_recall
    else:
        lexical = 0.25

    provider_prior = min(0.2, max(0.0, hit.score) * 0.2)
    raw_score = 0.65 * lexical + provider_prior + _authority_bonus(hit)
    return round(min(1.0, max(0.0, raw_score)), 4)


def select_urls_for_deep_fetch(
    scored_hits: Sequence[tuple[SearchHit, float]],
    *,
    max_deep_fetch: int,
    seed_urls: Sequence[str] = (),
    min_existing_content_chars: int = 320,
) -> list[SearchHit]:
    """Select only Top-K URLs that actually lack full body text or were user-seeded."""
    if max_deep_fetch <= 0:
        return []

    seed_canon = {canonicalize_url(url) for url in seed_urls if canonicalize_url(url)}
    selected: list[SearchHit] = []
    seen_ids: set[str] = set()

    # Priority 1: explicit seed URLs present in hits
    for hit, _score in scored_hits:
        cid = compute_canonical_id(url=hit.url, doi=hit.doi, arxiv_id=hit.arxiv_id)
        if canonicalize_url(hit.url) in seed_canon and cid not in seen_ids:
            seen_ids.add(cid)
            selected.append(hit)
            if len(selected) >= max_deep_fetch:
                return selected

    # Priority 2: top-scoring candidates whose snippet/content is too short for grounded synthesis
    ordered = sorted(scored_hits, key=lambda pair: pair[1], reverse=True)
    for hit, _score in ordered:
        cid = compute_canonical_id(url=hit.url, doi=hit.doi, arxiv_id=hit.arxiv_id)
        if cid in seen_ids:
            continue
        if len(hit.content.strip()) >= min_existing_content_chars:
            continue
        seen_ids.add(cid)
        selected.append(hit)
        if len(selected) >= max_deep_fetch:
            break

    return selected


def curate_evidence_batch(
    *,
    query: str,
    hits: Sequence[SearchHit],
    fetch_results: Sequence[FetchResult],
    seed_urls: Sequence[str] = (),
    top_k: int = 6,
    min_score: float = 0.05,
) -> CuratedEvidenceBatch:
    """Merge fetched pages into hits, filter noise, sanitize IPI, and Reverse-Repack."""
    deduped = deduplicate_hits(hits)
    seed_keys = {canonicalize_url(u) for u in seed_urls if canonicalize_url(u)}

    fetch_by_id: dict[str, FetchResult] = {}
    blocked_reports: list[BlockedSourceReport] = []
    seen_blocked_urls: set[str] = set()

    for res in fetch_results:
        cid = compute_canonical_id(
            url=res.url,
            doi=res.identifiers.doi,
            arxiv_id=res.identifiers.arxiv_id,
        )
        fetch_by_id[cid] = res
        fetch_by_id[canonicalize_url(res.url)] = res
        if res.is_blocked and res.blocked_reason is not None:
            canon_u = canonicalize_url(res.url) or res.url
            if canon_u not in seen_blocked_urls:
                seen_blocked_urls.add(canon_u)
                blocked_reports.append(
                    BlockedSourceReport(
                        url=res.url,
                        reason=res.blocked_reason,
                        doi=res.identifiers.doi,
                        arxiv_id=res.identifiers.arxiv_id,
                    )
                )

    candidates: list[tuple[SearchHit, float, bool]] = []
    represented_ids: set[str] = set()

    for hit in deduped:
        cid = compute_canonical_id(url=hit.url, doi=hit.doi, arxiv_id=hit.arxiv_id)
        url_key = canonicalize_url(hit.url)
        represented_ids.add(cid)
        represented_ids.add(url_key)

        fetched = fetch_by_id.get(cid) or fetch_by_id.get(url_key)
        enriched = hit
        is_direct_seed = url_key in seed_keys
        if fetched is not None and not fetched.is_blocked and fetched.markdown.strip():
            enriched = SearchHit(
                provider=hit.provider,
                category=hit.category,
                title=fetched.title or hit.title,
                url=fetched.final_url or hit.url,
                snippet=hit.snippet or fetched.markdown[:360],
                content=fetched.markdown,
                doi=hit.doi or fetched.identifiers.doi,
                arxiv_id=hit.arxiv_id or fetched.identifiers.arxiv_id,
                pmid=hit.pmid or fetched.identifiers.pmid,
                authors=hit.authors or fetched.authors,
                published_year=hit.published_year or fetched.published_year,
                open_access_pdf_url=hit.open_access_pdf_url or fetched.open_access_pdf_url,
                score=hit.score,
            )
            if fetched.stage_resolved == "client_supplied":
                is_direct_seed = True

        score = score_search_hit(query, enriched)
        candidates.append((enriched, score, is_direct_seed))

    # Promote standalone fetched URLs (e.g., direct_url mode or external_sources) not present in search hits
    for res in fetch_results:
        if res.is_blocked or not res.markdown.strip():
            continue
        cid = compute_canonical_id(
            url=res.url,
            doi=res.identifiers.doi,
            arxiv_id=res.identifiers.arxiv_id,
        )
        url_key = canonicalize_url(res.url)
        if cid in represented_ids or url_key in represented_ids:
            continue
        represented_ids.add(cid)
        represented_ids.add(url_key)

        derived = res.identifiers.merge(extract_academic_identifiers(res.url))
        is_scholar = bool(derived.doi or derived.arxiv_id or derived.pmid)
        synthetic = SearchHit(
            provider=res.stage_resolved or "direct_fetch",
            category="scholar" if is_scholar else "web",
            title=res.title or res.url,
            url=res.final_url or res.url,
            snippet=res.markdown[:360],
            content=res.markdown,
            doi=derived.doi,
            arxiv_id=derived.arxiv_id,
            pmid=derived.pmid,
            authors=res.authors,
            published_year=res.published_year,
            open_access_pdf_url=res.open_access_pdf_url,
            score=0.9,
        )
        score = max(0.5 if url_key in seed_keys or res.stage_resolved == "client_supplied" else 0.0, score_search_hit(query, synthetic))
        candidates.append((synthetic, score, True))

    filtered = [
        (hit, score)
        for hit, score, is_direct in candidates
        if score >= min_score or (is_direct and bool((hit.content or hit.snippet).strip()))
    ]
    filtered.sort(key=lambda pair: pair[1], reverse=True)
    top_pairs = filtered[: max(0, top_k)]

    ranked_list: list[RankedWebEvidence] = []
    for hit, score in top_pairs:
        source_id = compute_canonical_id(url=hit.url, doi=hit.doi, arxiv_id=hit.arxiv_id)
        body_raw = hit.content or hit.snippet
        sanitized = sanitize_untrusted_web_text(body_raw, source_url=hit.url) if body_raw else ""
        ranked_list.append(
            RankedWebEvidence(
                source_id=source_id,
                title=hit.title,
                url=hit.url,
                snippet=hit.snippet or body_raw[:280],
                content=sanitized,
                provider=hit.provider,
                category=hit.category,
                doi=hit.doi,
                arxiv_id=hit.arxiv_id,
                authors=hit.authors,
                published_year=hit.published_year,
                open_access_pdf_url=hit.open_access_pdf_url,
                score=score,
            )
        )

    ranked_desc = tuple(ranked_list)
    repacked = tuple(reversed(ranked_desc))
    return CuratedEvidenceBatch(
        repacked=repacked,
        ranked_desc=ranked_desc,
        blocked_sources=tuple(blocked_reports),
        total_raw_hits=len(hits),
    )
