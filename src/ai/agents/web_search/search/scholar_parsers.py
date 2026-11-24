"""Pure payload parsers shared by scholarly search providers and open-access mirror fetchers."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Final

ATOM_NS: Final[dict[str, str]] = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_ARXIV_ID_FROM_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^/?#\s]+)", re.IGNORECASE)
_DOI_PREFIX_RE = re.compile(r"^https?://(?:dx\.)?doi\.org/", re.IGNORECASE)
_MAX_AUTHORS: Final[int] = 10


@dataclass(frozen=True)
class ParsedScholarWork:
    """Normalized scholarly work metadata extracted from arXiv, OpenAlex, or Crossref payloads."""

    title: str
    abstract: str = ""
    url: str = ""
    doi: str | None = None
    arxiv_id: str | None = None
    pmid: str | None = None
    authors: tuple[str, ...] = ()
    published_year: int | None = None
    open_access_pdf_url: str | None = None


def strip_html_tags(raw: str) -> str:
    """Strip HTML/XML tags and collapse whitespace into clean single-spaced prose."""
    return " ".join(_HTML_TAG_RE.sub(" ", raw).split()).strip()


def normalize_doi(raw_doi: str | None) -> str | None:
    """Normalize a raw DOI or doi.org URL into lowercase bare DOI form."""
    if not raw_doi:
        return None
    cleaned = _DOI_PREFIX_RE.sub("", raw_doi.strip()).strip()
    return cleaned.lower() or None


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


def _extract_openalex_authors(raw_authorships: Any) -> tuple[str, ...]:
    if not isinstance(raw_authorships, list):
        return ()
    names: list[str] = []
    for entry in raw_authorships[:_MAX_AUTHORS]:
        if not isinstance(entry, dict):
            continue
        author = entry.get("author")
        if isinstance(author, dict):
            name = str(author.get("display_name") or "").strip()
            if name:
                names.append(name)
    return tuple(names)


def _extract_crossref_authors(raw_authors: Any) -> tuple[str, ...]:
    if not isinstance(raw_authors, list):
        return ()
    names: list[str] = []
    for entry in raw_authors[:_MAX_AUTHORS]:
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


def parse_arxiv_entry(
    entry: ET.Element,
    *,
    fallback_arxiv_id: str | None = None,
) -> ParsedScholarWork | None:
    """Parse one arXiv Atom `<entry>` element into normalized scholarly metadata."""
    title = " ".join(entry.findtext("atom:title", default="", namespaces=ATOM_NS).split())
    if not title:
        return None

    raw_url = entry.findtext("atom:id", default="", namespaces=ATOM_NS).strip()
    arxiv_match = _ARXIV_ID_FROM_URL_RE.search(raw_url)
    arxiv_id = (arxiv_match.group(1) if arxiv_match else None) or fallback_arxiv_id
    url = raw_url or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "")

    summary = " ".join(entry.findtext("atom:summary", default="", namespaces=ATOM_NS).split())
    authors = tuple(
        name
        for a in entry.findall("atom:author", ATOM_NS)[:_MAX_AUTHORS]
        if (name := " ".join(a.findtext("atom:name", default="", namespaces=ATOM_NS).split()))
    )
    published_raw = entry.findtext("atom:published", default="", namespaces=ATOM_NS).strip()
    year = (
        int(published_raw[:4]) if len(published_raw) >= 4 and published_raw[:4].isdigit() else None
    )
    doi = normalize_doi(entry.findtext("arxiv:doi", default="", namespaces=ATOM_NS))

    pdf_url: str | None = None
    for link in entry.findall("atom:link", ATOM_NS):
        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
            pdf_url = link.attrib.get("href")
            break
    if not pdf_url and arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

    return ParsedScholarWork(
        title=title,
        abstract=summary,
        url=url,
        doi=doi,
        arxiv_id=arxiv_id,
        authors=authors,
        published_year=year,
        open_access_pdf_url=pdf_url,
    )


def extract_openalex_work(item: dict[str, Any]) -> ParsedScholarWork | None:
    """Parse one OpenAlex work JSON object into normalized scholarly metadata."""
    title = str(item.get("title") or "").strip()
    if not title:
        return None

    doi = normalize_doi(item.get("doi") if isinstance(item.get("doi"), str) else None)
    primary_loc = item.get("primary_location")
    landing_url = (
        str(primary_loc.get("landing_page_url") or "").strip()
        if isinstance(primary_loc, dict)
        else ""
    )
    oa_info = item.get("open_access")
    oa_pdf = str(oa_info.get("oa_url") or "").strip() or None if isinstance(oa_info, dict) else None
    url = landing_url or (f"https://doi.org/{doi}" if doi else str(item.get("id") or ""))

    ids_obj = item.get("ids")
    pmid = (
        str(ids_obj.get("pmid") or "").rstrip("/").rsplit("/", 1)[-1] or None
        if isinstance(ids_obj, dict)
        else None
    )
    authors = _extract_openalex_authors(item.get("authorships"))
    abstract = reconstruct_openalex_abstract(item.get("abstract_inverted_index"))
    pub_year = item.get("publication_year")
    year = int(pub_year) if isinstance(pub_year, int) else None

    return ParsedScholarWork(
        title=title,
        abstract=abstract,
        url=url,
        doi=doi,
        pmid=pmid,
        authors=authors,
        published_year=year,
        open_access_pdf_url=oa_pdf,
    )


def extract_crossref_work(item: dict[str, Any]) -> ParsedScholarWork | None:
    """Parse one Crossref work JSON object into normalized scholarly metadata."""
    titles = item.get("title")
    title = str(titles[0]).strip() if isinstance(titles, list) and titles else ""
    if not title:
        return None

    doi = normalize_doi(item.get("DOI") if isinstance(item.get("DOI"), str) else None)
    url = str(item.get("URL") or (f"https://doi.org/{doi}" if doi else "")).strip()
    abstract = strip_html_tags(str(item.get("abstract") or ""))
    authors = _extract_crossref_authors(item.get("author"))
    year = _extract_crossref_year(item.get("issued"))

    return ParsedScholarWork(
        title=title,
        abstract=abstract,
        url=url,
        doi=doi,
        authors=authors,
        published_year=year,
    )
