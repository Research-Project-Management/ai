"""Scholarly `SearchProvider` adapters: OpenAlex, arXiv, and Crossref."""

import asyncio
import re
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from ai.agents.web_search.search.contracts import (
    ProviderCategory,
    SearchHit,
    SearchQuery,
)
from ai.agents.web_search.search.http import http_request
from ai.agents.web_search.search.scope import url_matches_domain_scope

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_ARXIV_ID_FROM_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^/?#\s]+)", re.IGNORECASE)
_DOI_PREFIX_RE = re.compile(r"^https?://(?:dx\.)?doi\.org/", re.IGNORECASE)
_ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def reconstruct_openalex_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Reassemble OpenAlex's inverted abstract word-position map into plain prose."""
    if not isinstance(inverted_index, dict) or not inverted_index:
        return ""
    tokens_by_pos: dict[int, str] = {}
    for word, positions in inverted_index.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int) and pos >= 0:
                tokens_by_pos[pos] = word
    if not tokens_by_pos:
        return ""
    return " ".join(tokens_by_pos[idx] for idx in sorted(tokens_by_pos))


def _strip_tags(raw: str) -> str:
    return " ".join(_HTML_TAG_RE.sub(" ", raw).split()).strip()


def _normalize_doi(raw_doi: str | None) -> str | None:
    if not raw_doi:
        return None
    cleaned = _DOI_PREFIX_RE.sub("", raw_doi.strip()).strip()
    return cleaned.lower() or None


class OpenAlexSearchProvider:
    """OpenAlex scholarly graph search returning structured DOI, authors, abstract & OA PDFs."""

    _SELECT_FIELDS = (
        "id,doi,title,publication_year,authorships,primary_location,open_access,"
        "abstract_inverted_index,ids"
    )

    def __init__(
        self,
        *,
        base_url: str = "https://api.openalex.org",
        api_key: str = "",
        timeout_seconds: float = 10.0,
        user_agent: str = "FluxResearchAI/0.1",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.strip().rstrip("/")
        self._api_key = api_key.strip()
        self._timeout_seconds = timeout_seconds
        self._user_agent = user_agent
        self._http_client = http_client
        self._sem = asyncio.Semaphore(1)

    @property
    def name(self) -> str:
        return "openalex"

    @property
    def category(self) -> ProviderCategory:
        return "scholar"

    async def search(self, query: SearchQuery) -> tuple[SearchHit, ...]:
        params: dict[str, Any] = {
            "search": query.query,
            "per-page": query.max_results,
            "select": self._SELECT_FIELDS,
        }
        if self._api_key:
            params["api_key"] = self._api_key

        async with self._sem:
            response = await http_request(
                method="GET",
                url=f"{self._base_url}/works",
                http_client=self._http_client,
                timeout_seconds=self._timeout_seconds,
                headers={"User-Agent": self._user_agent, "Accept": "application/json"},
                params=params,
            )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results") if isinstance(payload, dict) else []
        if not isinstance(results, list):
            return ()

        hits: list[SearchHit] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue

            doi = _normalize_doi(item.get("doi") if isinstance(item.get("doi"), str) else None)
            primary_loc = item.get("primary_location")
            landing_url = ""
            if isinstance(primary_loc, dict):
                landing_url = str(primary_loc.get("landing_page_url") or "").strip()
            oa_info = item.get("open_access")
            oa_pdf = (
                str(oa_info.get("oa_url") or "").strip() or None
                if isinstance(oa_info, dict)
                else None
            )
            url = landing_url or (f"https://doi.org/{doi}" if doi else str(item.get("id") or ""))
            if not url:
                continue
            if (query.allowed_domains or query.blocked_domains) and not url_matches_domain_scope(
                url,
                allowed_domains=query.allowed_domains,
                blocked_domains=query.blocked_domains,
            ):
                continue

            ids_obj = item.get("ids") if isinstance(item.get("ids"), dict) else {}
            pmid = (
                str(ids_obj.get("pmid") or "").rstrip("/").rsplit("/", 1)[-1] or None
                if isinstance(ids_obj, dict)
                else None
            )
            authors = _extract_openalex_authors(item.get("authorships"))
            abstract = reconstruct_openalex_abstract(item.get("abstract_inverted_index"))
            pub_year = item.get("publication_year")
            year = int(pub_year) if isinstance(pub_year, int) else None

            hits.append(
                SearchHit(
                    provider=self.name,
                    category=self.category,
                    title=title,
                    url=url,
                    snippet=abstract[:400],
                    content=abstract,
                    doi=doi,
                    pmid=pmid,
                    authors=authors,
                    published_year=year,
                    open_access_pdf_url=oa_pdf,
                    score=0.75 if abstract else 0.55,
                )
            )
        return tuple(hits)


def _extract_openalex_authors(raw_authorships: Any) -> tuple[str, ...]:
    if not isinstance(raw_authorships, list):
        return ()
    names: list[str] = []
    for entry in raw_authorships[:10]:
        if not isinstance(entry, dict):
            continue
        author = entry.get("author")
        if isinstance(author, dict):
            name = str(author.get("display_name") or "").strip()
            if name:
                names.append(name)
    return tuple(names)


class ArxivSearchProvider:
    """arXiv Atom XML search adapter returning preprints with clean abstracts and PDF URLs."""

    def __init__(
        self,
        *,
        base_url: str = "https://export.arxiv.org/api/query",
        timeout_seconds: float = 10.0,
        user_agent: str = "FluxResearchAI/0.1",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.strip().rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._user_agent = user_agent
        self._http_client = http_client
        self._sem = asyncio.Semaphore(1)

    @property
    def name(self) -> str:
        return "arxiv"

    @property
    def category(self) -> ProviderCategory:
        return "scholar"

    async def search(self, query: SearchQuery) -> tuple[SearchHit, ...]:
        async with self._sem:
            response = await http_request(
                method="GET",
                url=self._base_url,
                http_client=self._http_client,
                timeout_seconds=self._timeout_seconds,
                headers={"User-Agent": self._user_agent},
                params={
                    "search_query": f"all:{query.query}",
                    "start": 0,
                    "max_results": query.max_results,
                },
            )
        response.raise_for_status()
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError:
            return ()

        hits: list[SearchHit] = []
        for entry in root.findall("atom:entry", _ATOM_NS):
            title = " ".join(
                (entry.findtext("atom:title", default="", namespaces=_ATOM_NS)).split()
            )
            url = (entry.findtext("atom:id", default="", namespaces=_ATOM_NS)).strip()
            if not title or not url:
                continue
            if (query.allowed_domains or query.blocked_domains) and not url_matches_domain_scope(
                url,
                allowed_domains=query.allowed_domains,
                blocked_domains=query.blocked_domains,
            ):
                continue

            summary = " ".join(
                entry.findtext("atom:summary", default="", namespaces=_ATOM_NS).split()
            )
            authors = tuple(
                name
                for a in entry.findall("atom:author", _ATOM_NS)[:10]
                if (
                    name := " ".join(
                        a.findtext("atom:name", default="", namespaces=_ATOM_NS).split()
                    )
                )
            )
            published_raw = entry.findtext(
                "atom:published", default="", namespaces=_ATOM_NS
            ).strip()
            year = (
                int(published_raw[:4])
                if len(published_raw) >= 4 and published_raw[:4].isdigit()
                else None
            )
            doi = _normalize_doi(entry.findtext("arxiv:doi", default="", namespaces=_ATOM_NS))

            arxiv_match = _ARXIV_ID_FROM_URL_RE.search(url)
            arxiv_id = arxiv_match.group(1) if arxiv_match else None
            pdf_url: str | None = None
            for link in entry.findall("atom:link", _ATOM_NS):
                if (
                    link.attrib.get("title") == "pdf"
                    or link.attrib.get("type") == "application/pdf"
                ):
                    pdf_url = link.attrib.get("href")
                    break
            if not pdf_url and arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

            hits.append(
                SearchHit(
                    provider=self.name,
                    category=self.category,
                    title=title,
                    url=url,
                    snippet=summary[:400],
                    content=summary,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    authors=authors,
                    published_year=year,
                    open_access_pdf_url=pdf_url,
                    score=0.75 if summary else 0.55,
                )
            )
        return tuple(hits)


class CrossrefSearchProvider:
    """Crossref Works REST API adapter for publisher DOI & journal bibliographic records."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.crossref.org",
        timeout_seconds: float = 10.0,
        user_agent: str = "FluxResearchAI/0.1",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.strip().rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._user_agent = user_agent
        self._http_client = http_client
        self._sem = asyncio.Semaphore(1)

    @property
    def name(self) -> str:
        return "crossref"

    @property
    def category(self) -> ProviderCategory:
        return "scholar"

    async def search(self, query: SearchQuery) -> tuple[SearchHit, ...]:
        async with self._sem:
            response = await http_request(
                method="GET",
                url=f"{self._base_url}/works",
                http_client=self._http_client,
                timeout_seconds=self._timeout_seconds,
                headers={"User-Agent": self._user_agent, "Accept": "application/json"},
                params={"query": query.query, "rows": query.max_results},
            )
        response.raise_for_status()
        payload = response.json()
        message = payload.get("message") if isinstance(payload, dict) else {}
        items = message.get("items") if isinstance(message, dict) else []
        if not isinstance(items, list):
            return ()

        hits: list[SearchHit] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            titles = item.get("title")
            title = str(titles[0]).strip() if isinstance(titles, list) and titles else ""
            doi = _normalize_doi(item.get("DOI") if isinstance(item.get("DOI"), str) else None)
            url = str(item.get("URL") or (f"https://doi.org/{doi}" if doi else "")).strip()
            if not title or not url:
                continue
            if (query.allowed_domains or query.blocked_domains) and not url_matches_domain_scope(
                url,
                allowed_domains=query.allowed_domains,
                blocked_domains=query.blocked_domains,
            ):
                continue

            abstract = _strip_tags(str(item.get("abstract") or ""))
            authors = _extract_crossref_authors(item.get("author"))
            year = _extract_crossref_year(item.get("issued"))
            hits.append(
                SearchHit(
                    provider=self.name,
                    category=self.category,
                    title=title,
                    url=url,
                    snippet=abstract[:400],
                    content=abstract,
                    doi=doi,
                    authors=authors,
                    published_year=year,
                    score=0.65 if abstract else 0.5,
                )
            )
        return tuple(hits)


def _extract_crossref_authors(raw_authors: Any) -> tuple[str, ...]:
    if not isinstance(raw_authors, list):
        return ()
    names: list[str] = []
    for entry in raw_authors[:10]:
        if not isinstance(entry, dict):
            continue
        given = str(entry.get("given") or "").strip()
        family = str(entry.get("family") or "").strip()
        full = " ".join(part for part in (given, family) if part) or str(entry.get("name") or "")
        if full.strip():
            names.append(full.strip())
    return tuple(names)


def _extract_crossref_year(issued: Any) -> int | None:
    if not isinstance(issued, dict):
        return None
    parts = issued.get("date-parts")
    if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
        val = parts[0][0]
        if isinstance(val, int):
            return val
    return None
