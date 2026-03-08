"""Document upload and management endpoints."""

from __future__ import annotations

import logging
import os
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from app.models import DocumentInfo
from app.rag.ingestion import ingest_docx, ingest_pdf, ingest_text, ingest_xlsx

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "uploads")

# Extensions handled as plain UTF-8 text
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv",
    ".ts", ".tsx", ".js", ".jsx", ".mjs",
    ".json", ".py", ".html", ".css", ".xml", ".yaml", ".yml",
}
# Extensions we explicitly reject (binary, no text extraction available)
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}


@router.post("/upload", response_model=DocumentInfo)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(None),
    tags: str = Form(""),
):
    """
    Upload a document and ingest into the vector store.

    Supported formats:
    - PDF          → text extracted via pypdf
    - DOCX         → text extracted via python-docx (paragraphs + tables)
    - XLSX / XLS   → all sheets read via openpyxl
    - Plain text   → .txt, .md, .csv, .ts, .tsx, .js, .json, .py, …
    - Images       → rejected with 422 (no OCR available)
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Save uploaded file
    ext = os.path.splitext(file.filename or "doc")[1].lower()

    if ext in _IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Image files ({ext}) are not supported. "
                "Please upload PDF, DOCX, XLSX, or plain-text files."
            ),
        )

    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    metadata = {
        "title": title or file.filename or "Untitled",
        "tags": [t.strip() for t in tags.split(",") if t.strip()],
    }

    # Route to the appropriate ingester
    if ext == ".pdf":
        result = await ingest_pdf(file_path, metadata)
    elif ext == ".docx":
        result = await ingest_docx(file_path, metadata)
    elif ext in (".xlsx", ".xls"):
        result = await ingest_xlsx(file_path, metadata)
    elif ext in _TEXT_EXTENSIONS:
        text_content = content.decode("utf-8", errors="ignore")
        result = await ingest_text(text_content, file.filename or "upload", metadata)
    else:
        # Unknown extension — attempt UTF-8 text decode as best-effort
        logger.warning("Unknown extension %r — attempting plain-text ingestion", ext)
        text_content = content.decode("utf-8", errors="ignore")
        result = await ingest_text(text_content, file.filename or "upload", metadata)

    logger.info("Uploaded document: %s (%d chunks)", result["title"], result["chunk_count"])

    return DocumentInfo(**result)


@router.get("/bulk")
async def get_documents_bulk(
    ids: str = Query(..., description="Comma-separated document IDs"),
):
    """
    Return metadata (id, title, type) for the requested document IDs.

    Used by the frontend to resolve filenames for sessions restored from history.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    from app.config import get_settings
    from app.rag.store import get_qdrant_client

    doc_ids = [i.strip() for i in ids.split(",") if i.strip()]
    if not doc_ids:
        return []

    client = get_qdrant_client()
    settings = get_settings()

    results, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=Filter(
            must=[FieldCondition(key="metadata.doc_id", match=MatchAny(any=doc_ids))]
        ),
        limit=len(doc_ids) * 3,
        with_payload=True,
        with_vectors=False,
    )

    seen: dict[str, dict] = {}
    for point in results:
        payload = point.payload or {}
        meta = payload.get("metadata") or {}
        doc_id = meta.get("doc_id")
        if doc_id and doc_id not in seen:
            seen[doc_id] = {
                "id": doc_id,
                "title": meta.get("source") or meta.get("title") or "Untitled",
                "type": meta.get("type", "text"),
                "chunk_count": 0,
                "tags": meta.get("tags") or [],
            }

    return [seen[did] for did in doc_ids if did in seen]


@router.get("/", response_model=list[DocumentInfo])
async def list_documents():
    """List all ingested documents (upload directory listing)."""
    docs = []
    if os.path.exists(UPLOAD_DIR):
        for fname in os.listdir(UPLOAD_DIR):
            docs.append(
                DocumentInfo(
                    id=os.path.splitext(fname)[0],
                    title=fname,
                    chunk_count=0,
                    tags=[],
                )
            )
    return docs


@router.get("/{doc_id}")
async def get_document_content(doc_id: str):
    """
    Reconstruct and return the full text content of a document from Qdrant.

    Chunks are sorted by chunk_index and joined so the caller receives the
    document in its original reading order.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    from app.config import get_settings
    from app.rag.store import get_qdrant_client

    client = get_qdrant_client()
    settings = get_settings()

    results, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=Filter(
            must=[FieldCondition(key="metadata.doc_id", match=MatchValue(value=doc_id))]
        ),
        limit=500,
        with_payload=True,
        with_vectors=False,
    )

    if not results:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks = sorted(
        results,
        key=lambda p: (p.payload or {}).get("metadata", {}).get("chunk_index", 0),
    )

    first_meta = (chunks[0].payload or {}).get("metadata") or {}
    content = "\n\n".join(
        (p.payload or {}).get("page_content", "")
        for p in chunks
        if (p.payload or {}).get("page_content")
    )

    return {
        "id": doc_id,
        "title": first_meta.get("source") or first_meta.get("title") or "Untitled",
        "type": first_meta.get("type", "text"),
        "content": content,
        "chunk_count": len(chunks),
        "tags": first_meta.get("tags") or [],
    }
