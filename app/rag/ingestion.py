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


async def ingest_pdf(file_path: str, metadata: dict[str, Any] | None = None) -> dict:
    """
    Ingest a PDF file: parse → chunk → embed → store in Qdrant.

    Returns info about the ingested document.
    """
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)

    doc_id = str(uuid.uuid4())
    base_metadata = {
        "doc_id": doc_id,
        "source": Path(file_path).name,
        "type": "pdf",
        **(metadata or {}),
    }

    return await _ingest_text(text, base_metadata, doc_id)


async def ingest_text(
    text: str,
    source_name: str = "manual",
    metadata: dict[str, Any] | None = None,
) -> dict:
    """Ingest raw text content."""
    doc_id = str(uuid.uuid4())
    base_metadata = {
        "doc_id": doc_id,
        "source": source_name,
        "type": "text",
        **(metadata or {}),
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
