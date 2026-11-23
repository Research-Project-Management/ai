"""Deterministic URL, academic identifier, and clean HTML-to-Markdown extractors."""

import re
from html.parser import HTMLParser
from typing import ClassVar

from ai.agents.web_search.fetch.contracts import AcademicIdentifiers

_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_DOI_URL_RE = re.compile(
    r"(?:doi\.org/|springer\.com/(?:article|chapter|book)/|wiley\.com/doi/(?:abs/|full/|pdf/|epdf/)?|dl\.acm\.org/doi/(?:abs/|full/|pdf/)?|tandfonline\.com/doi/(?:abs/|full/)?)(10\.\d{4,9}/[^\s?#\"'<>]+)",
    re.IGNORECASE,
)
_RAW_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b")
_ARXIV_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf|html)/|arxiv:)\s*(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)
_PMID_RE = re.compile(
    r"(?:pubmed\.ncbi\.nlm\.nih\.gov/|europepmc\.org/article/MED/|pmid:\s*)(\d{4,10})",
    re.IGNORECASE,
)
_PMCID_RE = re.compile(
    r"(?:ncbi\.nlm\.nih\.gov/pmc/articles/|pmc\.ncbi\.nlm\.nih\.gov/articles/|europepmc\.org/article/PMC/|\b)(PMC\d{4,10})",
    re.IGNORECASE,
)


def extract_urls_from_text(text: str) -> tuple[str, ...]:
    """Extract normalized HTTP(S) URLs from user messages without trailing punctuation."""
    urls: list[str] = []
    seen: set[str] = set()
    for match in _URL_RE.finditer(text):
        cleaned = match.group(0).rstrip(".,;:!?)]}>\"'")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            urls.append(cleaned)
    return tuple(urls)


def extract_academic_identifiers(text_or_url: str) -> AcademicIdentifiers:
    """Derive DOI, arXiv ID, PMID, and PMCID from URLs or query strings (0ms, offline)."""
    text = text_or_url.strip()
    if not text:
        return AcademicIdentifiers()

    doi_match = _DOI_URL_RE.search(text) or _RAW_DOI_RE.search(text)
    doi: str | None = None
    if doi_match:
        doi = doi_match.group(1).rstrip(".,;:!?)]}>\"'").lower().removesuffix(".pdf")

    arxiv_match = _ARXIV_RE.search(text)
    arxiv_id = arxiv_match.group(1).removesuffix(".pdf") if arxiv_match else None

    pmid_match = _PMID_RE.search(text)
    pmid = pmid_match.group(1) if pmid_match else None

    pmcid_match = _PMCID_RE.search(text)
    pmcid = pmcid_match.group(1).upper() if pmcid_match else None

    return AcademicIdentifiers(doi=doi, arxiv_id=arxiv_id, pmid=pmid, pmcid=pmcid)


class _MarkdownHtmlParser(HTMLParser):
    """Lightweight stdlib HTML-to-Markdown extractor stripping nav/script/boilerplate."""

    _SKIP_TAGS: ClassVar[frozenset[str]] = frozenset(
        {"script", "style", "noscript", "svg", "iframe", "nav", "footer", "header", "aside", "form"}
    )
    _HEADING_MAP: ClassVar[dict[str, str]] = {
        "h1": "# ",
        "h2": "## ",
        "h3": "### ",
        "h4": "#### ",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self._title_parts: list[str] = []
        self._chunks: list[str] = []
        self._in_table_cell = False
        self._current_cell: list[str] = []
        self._current_row: list[str] = []

    @property
    def title(self) -> str:
        return " ".join("".join(self._title_parts).split()).strip()

    @property
    def markdown(self) -> str:
        raw = "".join(self._chunks)
        lines = [
            " ".join(line.split()) if not line.startswith("|") else line
            for line in raw.splitlines()
        ]
        compacted: list[str] = []
        blank_streak = 0
        for line in lines:
            if not line.strip():
                blank_streak += 1
                if blank_streak <= 1 and compacted:
                    compacted.append("")
            else:
                blank_streak = 0
                compacted.append(line.strip())
        return "\n".join(compacted).strip()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        low = tag.lower()
        if low in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        if low == "title":
            self._in_title = True
            return
        if low in self._HEADING_MAP:
            self._chunks.append(f"\n\n{self._HEADING_MAP[low]}")
        elif low == "li":
            self._chunks.append("\n- ")
        elif low in {"p", "br", "div", "section", "article", "blockquote", "pre"}:
            self._chunks.append("\n")
        elif low == "tr":
            self._current_row = []
        elif low in {"td", "th"}:
            self._in_table_cell = True
            self._current_cell = []

    def handle_endtag(self, tag: str) -> None:
        low = tag.lower()
        if low in self._SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if self._skip_depth > 0:
            return
        if low == "title":
            self._in_title = False
            return
        if low in {"h1", "h2", "h3", "h4", "p", "blockquote", "pre", "li"}:
            self._chunks.append("\n")
        elif low in {"td", "th"} and self._in_table_cell:
            self._in_table_cell = False
            cell_text = " ".join("".join(self._current_cell).split())
            self._current_row.append(cell_text)
            self._current_cell = []
        elif low == "tr" and self._current_row:
            self._chunks.append("\n| " + " | ".join(self._current_row) + " |")
            self._current_row = []
        elif low == "table":
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title:
            self._title_parts.append(data)
            return
        if self._in_table_cell:
            self._current_cell.append(data)
            return
        self._chunks.append(data)


def extract_markdown_from_html(raw_html: str, *, max_chars: int = 12_000) -> tuple[str, str]:
    """Convert raw HTML into `(title, clean_markdown)` capped to `max_chars`."""
    parser = _MarkdownHtmlParser()
    parser.feed(raw_html)
    parser.close()
    markdown = parser.markdown
    if len(markdown) > max_chars:
        markdown = markdown[:max_chars].rstrip() + "..."
    return parser.title, markdown
