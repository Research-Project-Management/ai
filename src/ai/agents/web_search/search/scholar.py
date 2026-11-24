"""Scholarly `SearchProvider` adapters: OpenAlex, arXiv, and Crossref."""

import asyncio
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from ai.agents.web_search.search.contracts import (
    ProviderCategory,
    SearchHit,
    SearchQuery,
)
from ai.agents.web_search.search.http import http_request
from ai.agents.web_search.search.scholar_parsers import (
    ATOM_NS,
    extract_crossref_work,
    extract_openalex_work,
    parse_arxiv_entry,
)
from ai.agents.web_search.search.scope import url_matches_domain_scope


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
            work = extract_openalex_work(item)
            if work is None or not work.url:
                continue
            if (query.allowed_domains or query.blocked_domains) and not url_matches_domain_scope(
                work.url,
                allowed_domains=query.allowed_domains,
                blocked_domains=query.blocked_domains,
            ):
                continue

            hits.append(
                SearchHit(
                    provider=self.name,
                    category=self.category,
                    title=work.title,
                    url=work.url,
                    snippet=work.abstract[:400],
                    content=work.abstract,
                    doi=work.doi,
                    pmid=work.pmid,
                    authors=work.authors,
                    published_year=work.published_year,
                    open_access_pdf_url=work.open_access_pdf_url,
                    score=0.75 if work.abstract else 0.55,
                )
            )
        return tuple(hits)


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
        for entry in root.findall("atom:entry", ATOM_NS):
            work = parse_arxiv_entry(entry)
            if work is None or not work.url:
                continue
            if (query.allowed_domains or query.blocked_domains) and not url_matches_domain_scope(
                work.url,
                allowed_domains=query.allowed_domains,
                blocked_domains=query.blocked_domains,
            ):
                continue

            hits.append(
                SearchHit(
                    provider=self.name,
                    category=self.category,
                    title=work.title,
                    url=work.url,
                    snippet=work.abstract[:400],
                    content=work.abstract,
                    doi=work.doi,
                    arxiv_id=work.arxiv_id,
                    authors=work.authors,
                    published_year=work.published_year,
                    open_access_pdf_url=work.open_access_pdf_url,
                    score=0.75 if work.abstract else 0.55,
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
            work = extract_crossref_work(item)
            if work is None or not work.url:
                continue
            if (query.allowed_domains or query.blocked_domains) and not url_matches_domain_scope(
                work.url,
                allowed_domains=query.allowed_domains,
                blocked_domains=query.blocked_domains,
            ):
                continue

            hits.append(
                SearchHit(
                    provider=self.name,
                    category=self.category,
                    title=work.title,
                    url=work.url,
                    snippet=work.abstract[:400],
                    content=work.abstract,
                    doi=work.doi,
                    authors=work.authors,
                    published_year=work.published_year,
                    score=0.65 if work.abstract else 0.5,
                )
            )
        return tuple(hits)
