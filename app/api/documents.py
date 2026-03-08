"""Document upload and management endpoints."""

from __future__ import annotations

import logging
import os
import uuid

from fastapi import APIRouter, File, Form, UploadFile

from app.models import DocumentInfo
from app.rag.ingestion import ingest_pdf, ingest_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")


@router.post("/upload", response_model=DocumentInfo)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(None),
    tags: str = Form(""),
):
    """
    Upload a document (PDF or text) and ingest into the vector store.

    The document is chunked, embedded, and stored in Qdrant for
    later retrieval by the RAG agent.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Save uploaded file
    ext = os.path.splitext(file.filename or "doc")[1].lower()
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    metadata = {
        "title": title or file.filename or "Untitled",
        "tags": [t.strip() for t in tags.split(",") if t.strip()],
    }

    # Ingest based on file type
    if ext == ".pdf":
        result = await ingest_pdf(file_path, metadata)
    else:
        text_content = content.decode("utf-8", errors="ignore")
        result = await ingest_text(text_content, file.filename or "upload", metadata)

    logger.info("Uploaded document: %s (%d chunks)", result["title"], result["chunk_count"])

    return DocumentInfo(**result)


@router.get("/", response_model=list[DocumentInfo])
async def list_documents():
    """List all ingested documents."""
    # For MVP, list files in upload dir
    docs = []
    if os.path.exists(UPLOAD_DIR):
        for fname in os.listdir(UPLOAD_DIR):
            docs.append(
                DocumentInfo(
                    id=os.path.splitext(fname)[0],
                    title=fname,
                    chunk_count=0,  # Would need DB lookup for accurate count
                    tags=[],
                )
            )
    return docs
