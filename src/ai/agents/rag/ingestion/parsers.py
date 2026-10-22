"""Format-specific DocumentParsers (PDF, DOCX, Markdown/Text) and ParserRegistry."""

import io
import re
from pathlib import Path
from typing import Any

import pymupdf
import pymupdf4llm
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from ai.agents.rag.ingestion.contracts import (
    DocumentParser,
    ParsedDocument,
    ParsedPage,
    UnsupportedDocumentFormatError,
)

_H1_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)
_HEADING_STYLE_RE = re.compile(r"^Heading\s+(\d)$", re.IGNORECASE)


def _extract_markdown_title(text: str) -> str:
    match = _H1_RE.search(text)
    return match.group(1).strip() if match else ""


class MarkdownTextParser:
    """Parser for UTF-8 Markdown and plain-text documents."""

    supported_extensions: frozenset[str] = frozenset({".md", ".markdown", ".txt"})
    supported_mime_types: frozenset[str] = frozenset({"text/markdown", "text/plain", "text/x-markdown"})

    def parse_bytes(self, data: bytes, *, filename: str = "") -> ParsedDocument:
        text = data.decode("utf-8", errors="replace").strip()
        title = _extract_markdown_title(text) or Path(filename).stem
        pages = (ParsedPage(page_number=1, markdown=text),) if text else ()
        return ParsedDocument(title=title, pages=pages)


class PdfMarkdownParser:
    """Layout-aware PDF parser preserving page numbers, headings, and Markdown tables."""

    supported_extensions: frozenset[str] = frozenset({".pdf"})
    supported_mime_types: frozenset[str] = frozenset({"application/pdf"})

    def parse_bytes(self, data: bytes, *, filename: str = "") -> ParsedDocument:
        doc = pymupdf.open(stream=data, filetype="pdf")
        try:
            meta: dict[str, Any] = dict(doc.metadata or {})
            raw_chunks = pymupdf4llm.to_markdown(doc, page_chunks=True)
            pages: list[ParsedPage] = []

            if isinstance(raw_chunks, list) and raw_chunks:
                for idx, chunk in enumerate(raw_chunks, start=1):
                    if not isinstance(chunk, dict):
                        continue
                    chunk_meta = chunk.get("metadata")
                    page_num = idx
                    if isinstance(chunk_meta, dict) and chunk_meta.get("page") is not None:
                        page_num = int(chunk_meta["page"])
                    md_text = str(chunk.get("text") or "").strip()
                    if not md_text and 1 <= page_num <= doc.page_count:
                        md_text = str(doc.load_page(page_num - 1).get_text("text") or "").strip()
                    if md_text:
                        pages.append(ParsedPage(page_number=page_num, markdown=md_text))
            else:
                for page_idx in range(doc.page_count):
                    txt = str(doc.load_page(page_idx).get_text("text") or "").strip()
                    if txt:
                        pages.append(ParsedPage(page_number=page_idx + 1, markdown=txt))

            title = str(meta.get("title") or "").strip()
            if not title and pages:
                title = _extract_markdown_title(pages[0].markdown) or Path(filename).stem

            cleaned_meta = {k: v for k, v in meta.items() if isinstance(v, str) and v.strip()}
            return ParsedDocument(title=title, pages=tuple(pages), metadata=cleaned_meta)
        finally:
            doc.close()


def _format_docx_table(table: Table) -> str:
    rows = table.rows
    if not rows:
        return ""
    matrix: list[list[str]] = []
    for row in rows:
        cells = [" ".join(cell.text.split()).replace("|", "\\|") for cell in row.cells]
        matrix.append(cells)

    col_count = max(len(r) for r in matrix)
    if col_count == 0:
        return ""

    normalized = [r + [""] * (col_count - len(r)) for r in matrix]
    header = "| " + " | ".join(normalized[0]) + " |"
    separator = "| " + " | ".join("---" for _ in range(col_count)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in normalized[1:]]
    return "\n".join([header, separator, *body])


class DocxMarkdownParser:
    """Parser converting `.docx` paragraphs, headings, and tables into Markdown."""

    supported_extensions: frozenset[str] = frozenset({".docx"})
    supported_mime_types: frozenset[str] = frozenset(
        {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    )

    def parse_bytes(self, data: bytes, *, filename: str = "") -> ParsedDocument:
        doc = Document(io.BytesIO(data))
        blocks: list[str] = []
        inferred_title = ""

        for item in doc.iter_inner_content():
            if isinstance(item, Paragraph):
                text = item.text.strip()
                if not text:
                    continue
                style_name = str(item.style.name if item.style is not None else "")
                if style_name.lower() == "title":
                    inferred_title = inferred_title or text
                    blocks.append(f"# {text}")
                else:
                    match = _HEADING_STYLE_RE.match(style_name)
                    if match:
                        level = min(4, max(1, int(match.group(1))))
                        if level == 1 and not inferred_title:
                            inferred_title = text
                        blocks.append(f"{'#' * level} {text}")
                    else:
                        blocks.append(text)
            elif isinstance(item, Table):
                md_table = _format_docx_table(item)
                if md_table:
                    blocks.append(md_table)

        markdown = "\n\n".join(blocks).strip()
        prop_title = str(doc.core_properties.title or "").strip()
        title = prop_title or inferred_title or Path(filename).stem
        pages = (ParsedPage(page_number=1, markdown=markdown),) if markdown else ()
        return ParsedDocument(title=title, pages=pages)


class ParserRegistry:
    """OCP registry mapping file extensions and MIME types to DocumentParsers."""

    def __init__(self, parsers: tuple[DocumentParser, ...] = ()) -> None:
        self._by_ext: dict[str, DocumentParser] = {}
        self._by_mime: dict[str, DocumentParser] = {}
        for parser in parsers:
            self.register(parser)

    def register(self, parser: DocumentParser) -> None:
        for ext in parser.supported_extensions:
            self._by_ext[ext.lower()] = parser
        for mime in parser.supported_mime_types:
            self._by_mime[mime.lower()] = parser

    def resolve(self, *, filename: str = "", mime_type: str = "") -> DocumentParser:
        ext = Path(filename).suffix.lower() if filename else ""
        if ext and ext in self._by_ext:
            return self._by_ext[ext]

        normalized_mime = mime_type.split(";")[0].strip().lower()
        if normalized_mime and normalized_mime in self._by_mime:
            return self._by_mime[normalized_mime]

        raise UnsupportedDocumentFormatError(
            f"Unsupported document format (filename={filename!r}, mime_type={mime_type!r})"
        )


def build_default_parser_registry() -> ParserRegistry:
    """Assemble the default parser registry for Markdown/TXT, PDF, and DOCX."""
    return ParserRegistry(
        (
            MarkdownTextParser(),
            PdfMarkdownParser(),
            DocxMarkdownParser(),
        )
    )
