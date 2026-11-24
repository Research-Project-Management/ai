"""Fetch stage resolving open-access copies from arXiv, Europe PMC, OpenAlex, and Crossref."""

import xml.etree.ElementTree as ET
from typing import Any

import httpx

from ai.agents.web_search.fetch.contracts import (
    AcademicIdentifiers,
    FetchRequest,
    FetchResult,
)
from ai.agents.web_search.search.scholar_parsers import (
    ATOM_NS,
    extract_crossref_work,
    extract_openalex_work,
    parse_arxiv_entry,
    strip_html_tags,
)


class OpenAcademicMirrorStage:
    """Stage 2: Resolve DOI / arXiv / PMID via public scholarly APIs, bypassing publisher WAFs."""

    def __init__(
        self,
        *,
        openalex_base_url: str = "https://api.openalex.org",
        openalex_api_key: str = "",
        arxiv_base_url: str = "https://export.arxiv.org/api/query",
        europepmc_base_url: str = "https://www.ebi.ac.uk/europepmc/webservices/rest",
        crossref_base_url: str = "https://api.crossref.org",
        timeout_seconds: float = 8.0,
        user_agent: str = "FluxResearchAI/0.1",
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._openalex_base_url = openalex_base_url.strip().rstrip("/")
        self._openalex_api_key = openalex_api_key.strip()
        self._arxiv_base_url = arxiv_base_url.strip().rstrip("/")
        self._europepmc_base_url = europepmc_base_url.strip().rstrip("/")
        self._crossref_base_url = crossref_base_url.strip().rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._user_agent = user_agent
        self._http_client = http_client

    @property
    def name(self) -> str:
        return "open_mirror"

    async def _get(self, url: str, *, params: dict[str, Any] | None = None) -> httpx.Response:
        headers = {"User-Agent": self._user_agent, "Accept": "application/json,application/xml"}
        if self._http_client is not None:
            return await self._http_client.get(
                url, params=params, headers=headers, timeout=self._timeout_seconds
            )
        async with httpx.AsyncClient(
            timeout=self._timeout_seconds, follow_redirects=True
        ) as client:
            return await client.get(url, params=params, headers=headers)

    async def try_fetch(self, request: FetchRequest) -> FetchResult | None:
        ids = request.identifiers
        if not ids.has_any:
            return None

        if ids.arxiv_id:
            result = await self._resolve_arxiv(request.url, ids)
            if result is not None:
                return result

        if ids.pmid or ids.pmcid or ids.doi:
            result = await self._resolve_europepmc(request.url, ids)
            if result is not None:
                return result

        if ids.doi:
            for resolver in (self._resolve_openalex_doi, self._resolve_crossref_doi):
                result = await resolver(request.url, ids)
                if result is not None:
                    return result

        return None

    async def _resolve_arxiv(
        self, original_url: str, ids: AcademicIdentifiers
    ) -> FetchResult | None:
        if not ids.arxiv_id:
            return None
        try:
            resp = await self._get(self._arxiv_base_url, params={"id_list": ids.arxiv_id})
            if resp.status_code != 200:
                return None
            root = ET.fromstring(resp.text)
        except (httpx.HTTPError, ET.ParseError, ValueError):
            return None

        entry = root.find("atom:entry", ATOM_NS)
        if entry is None:
            return None
        work = parse_arxiv_entry(entry, fallback_arxiv_id=ids.arxiv_id)
        if work is None or not work.abstract:
            return None

        arxiv_url = f"https://arxiv.org/abs/{ids.arxiv_id}"
        pdf_url = work.open_access_pdf_url or f"https://arxiv.org/pdf/{ids.arxiv_id}"
        markdown = (
            f"# {work.title}\n\nAuthors: {', '.join(work.authors)}\n\n## Abstract\n{work.abstract}"
        )
        return FetchResult(
            url=original_url,
            final_url=arxiv_url,
            title=work.title,
            markdown=markdown,
            stage_resolved=self.name,
            identifiers=ids,
            authors=work.authors,
            published_year=work.published_year,
            open_access_pdf_url=pdf_url,
        )

    async def _resolve_europepmc(
        self, original_url: str, ids: AcademicIdentifiers
    ) -> FetchResult | None:
        raw_query = (
            f'DOI:"{ids.doi}"'
            if ids.doi
            else f"EXT_ID:{ids.pmid}"
            if ids.pmid
            else f"PMC:{ids.pmcid}"
        )
        try:
            resp = await self._get(
                f"{self._europepmc_base_url}/search",
                params={"query": raw_query, "format": "json", "resultType": "core"},
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
        except (httpx.HTTPError, ValueError):
            return None

        records = (data.get("resultList") or {}).get("result") if isinstance(data, dict) else None
        if not isinstance(records, list) or not records or not isinstance(records[0], dict):
            return None
        rec = records[0]
        title = str(rec.get("title") or "").strip()
        abstract = strip_html_tags(str(rec.get("abstractText") or ""))
        if not title or not abstract:
            return None

        author_str = str(rec.get("authorString") or "").strip()
        authors = tuple(a.strip() for a in author_str.split(",")[:10] if a.strip())
        pub_year_str = str(rec.get("pubYear") or "").strip()
        year = int(pub_year_str) if pub_year_str.isdigit() else None
        doi_val = str(rec.get("doi") or ids.doi or "").strip().lower() or None
        markdown = f"# {title}\n\n## Abstract\n{abstract}"
        return FetchResult(
            url=original_url,
            final_url=f"https://doi.org/{doi_val}" if doi_val else original_url,
            title=title,
            markdown=markdown,
            stage_resolved=self.name,
            identifiers=ids.merge(AcademicIdentifiers(doi=doi_val)),
            authors=authors,
            published_year=year,
        )

    async def _resolve_openalex_doi(
        self, original_url: str, ids: AcademicIdentifiers
    ) -> FetchResult | None:
        if not ids.doi:
            return None
        params: dict[str, Any] = {}
        if self._openalex_api_key:
            params["api_key"] = self._openalex_api_key
        try:
            resp = await self._get(
                f"{self._openalex_base_url}/works/https://doi.org/{ids.doi}",
                params=params or None,
            )
            if resp.status_code != 200:
                return None
            rec = resp.json()
        except (httpx.HTTPError, ValueError):
            return None

        if not isinstance(rec, dict):
            return None
        work = extract_openalex_work(rec)
        if work is None or not work.abstract:
            return None

        markdown = f"# {work.title}\n\n## Abstract\n{work.abstract}"
        return FetchResult(
            url=original_url,
            final_url=f"https://doi.org/{ids.doi}",
            title=work.title,
            markdown=markdown,
            stage_resolved=self.name,
            identifiers=ids,
            authors=work.authors,
            published_year=work.published_year,
            open_access_pdf_url=work.open_access_pdf_url,
        )

    async def _resolve_crossref_doi(
        self, original_url: str, ids: AcademicIdentifiers
    ) -> FetchResult | None:
        if not ids.doi:
            return None
        try:
            resp = await self._get(f"{self._crossref_base_url}/works/{ids.doi}")
            if resp.status_code != 200:
                return None
            payload = resp.json()
        except (httpx.HTTPError, ValueError):
            return None

        msg = payload.get("message") if isinstance(payload, dict) else None
        if not isinstance(msg, dict):
            return None
        work = extract_crossref_work(msg)
        if work is None or not work.abstract:
            return None

        markdown = f"# {work.title}\n\n## Abstract\n{work.abstract}"
        return FetchResult(
            url=original_url,
            final_url=work.url or f"https://doi.org/{ids.doi}",
            title=work.title,
            markdown=markdown,
            stage_resolved=self.name,
            identifiers=ids,
            authors=work.authors,
            published_year=work.published_year,
        )
