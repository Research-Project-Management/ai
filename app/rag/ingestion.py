"""Document ingestion pipeline — parse, chunk, and store documents."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.store import get_vector_store

logger = logging.getLogger(__name__)

# Default chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


async def ingest_pdf(
    file_path: str,
    metadata: dict[str, Any] | None = None,
    user_id: str = "",
    chat_id: str = "",
) -> dict:
    """
    Ingest a PDF file: parse → chunk → embed → store in Qdrant.

    Returns info about the ingested document.
    """
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)

    doc_id = str(uuid.uuid4())
    # Use original filename from metadata if available; fall back to the
    # UUID-based saved filename only when no title was supplied.
    source_name = (metadata or {}).get("title") or Path(file_path).name
    base_metadata = {
        "doc_id": doc_id,
        "source": source_name,
        "type": "pdf",
        **(metadata or {}),
        # Always override — must not be controllable by caller
        "user_id": user_id,
        "chat_id": chat_id,
    }

    return await _ingest_text(text, base_metadata, doc_id)


async def ingest_docx(
    file_path: str,
    metadata: dict[str, Any] | None = None,
    user_id: str = "",
    chat_id: str = "",
) -> dict:
    """
    Ingest a Word document (.docx): extract paragraphs + tables → chunk → store.

    Returns info about the ingested document.
    """
    from docx import Document as DocxDocument

    doc = DocxDocument(file_path)

    parts: list[str] = []

    # Body paragraphs
    for para in doc.paragraphs:
        stripped = para.text.strip()
        if stripped:
            parts.append(stripped)

    # Tables (each row as pipe-delimited text)
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    text = "\n\n".join(parts)

    doc_id = str(uuid.uuid4())
    source_name = (metadata or {}).get("title") or Path(file_path).name
    base_metadata = {
        "doc_id": doc_id,
        "source": source_name,
        "type": "docx",
        **(metadata or {}),
        # Always override — must not be controllable by caller
        "user_id": user_id,
        "chat_id": chat_id,
    }
    return await _ingest_text(text, base_metadata, doc_id)


async def ingest_xlsx(
    file_path: str,
    metadata: dict[str, Any] | None = None,
    user_id: str = "",
    chat_id: str = "",
) -> dict:
    """
    Ingest an Excel spreadsheet (.xlsx/.xls): read all sheets → chunk → store.

    Returns info about the ingested document.
    """
    from openpyxl import load_workbook

    wb = load_workbook(file_path, read_only=True, data_only=True)
    lines: list[str] = []
    for sheet in wb.worksheets:
        lines.append(f"## Sheet: {sheet.title}")
        for row in sheet.iter_rows(values_only=True):
            row_cells = [str(v) for v in row if v is not None and str(v).strip()]
            if row_cells:
                lines.append(" | ".join(row_cells))
    wb.close()

    text = "\n".join(lines)

    doc_id = str(uuid.uuid4())
    source_name = (metadata or {}).get("title") or Path(file_path).name
    base_metadata = {
        "doc_id": doc_id,
        "source": source_name,
        "type": "xlsx",
        **(metadata or {}),
        # Always override — must not be controllable by caller
        "user_id": user_id,
        "chat_id": chat_id,
    }
    return await _ingest_text(text, base_metadata, doc_id)


async def ingest_text(
    text: str,
    source_name: str = "manual",
    metadata: dict[str, Any] | None = None,
    user_id: str = "",
    chat_id: str = "",
) -> dict:
    """Ingest raw text content."""
    doc_id = str(uuid.uuid4())
    base_metadata = {
        "doc_id": doc_id,
        "source": source_name,
        "type": "text",
        **(metadata or {}),
        # Always override — must not be controllable by caller
        "user_id": user_id,
        "chat_id": chat_id,
    }
    return await _ingest_text(text, base_metadata, doc_id)


async def _ingest_text(text: str, metadata: dict, doc_id: str) -> dict:
    """Internal: chunk text and store in vector DB."""
    chunks = splitter.split_text(text)

    documents = [
        Document(
            page_content=chunk,
            metadata={**metadata, "chunk_index": i},
        )
        for i, chunk in enumerate(chunks)
    ]

    store = get_vector_store()
    await store.aadd_documents(documents)

    logger.info(
        "Ingested doc %s (%s): %d chunks",
        doc_id,
        metadata.get("source", "unknown"),
        len(chunks),
    )

    return {
        "id": doc_id,
        "title": metadata.get("title", metadata.get("source", "Untitled")),
        "chunk_count": len(chunks),
        "tags": metadata.get("tags", []),
    }
