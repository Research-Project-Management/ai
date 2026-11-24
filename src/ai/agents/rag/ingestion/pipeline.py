"""Write-side ingestion pipeline orchestrating parsing, chunking, and upsert."""

import asyncio

from ai.agents.rag.ingestion.chunker import MarkdownStructureChunker
from ai.agents.rag.ingestion.contracts import (
    IngestDocumentRequest,
    IngestResult,
    ParsedDocument,
    ParsedPage,
)
from ai.agents.rag.ingestion.parsers import ParserRegistry
from ai.shared.contracts.knowledge import (
    KnowledgeChunkInput,
    KnowledgeDocumentInput,
    KnowledgeStore,
)


class IngestionPipeline:
    """Deep module translating raw files or Markdown text into indexed Qdrant vectors."""

    def __init__(
        self,
        *,
        parser_registry: ParserRegistry,
        chunker: MarkdownStructureChunker,
        knowledge_store: KnowledgeStore,
        max_file_bytes: int = 50 * 1024 * 1024,
    ) -> None:
        self._parser_registry = parser_registry
        self._chunker = chunker
        self._knowledge_store = knowledge_store
        self._max_file_bytes = max_file_bytes

    @property
    def max_file_bytes(self) -> int:
        return self._max_file_bytes

    def _parse_and_chunk(
        self,
        request: IngestDocumentRequest,
    ) -> tuple[str, int, list[KnowledgeChunkInput]]:
        if request.raw_bytes is not None:
            parser = self._parser_registry.resolve(
                filename=request.filename,
                mime_type=request.mime_type,
            )
            parsed = parser.parse_bytes(request.raw_bytes, filename=request.filename)
        else:
            parsed = ParsedDocument(
                title=request.title,
                pages=(ParsedPage(page_number=None, markdown=request.content),),
            )

        resolved_title = (request.title or parsed.title or request.filename).strip()
        chunks = self._chunker.chunk(parsed, title=resolved_title)
        return resolved_title, len(parsed.pages), chunks

    async def ingest(self, request: IngestDocumentRequest) -> IngestResult:
        """Parse, structure-chunk in worker thread, and idempotently upsert into KnowledgeStore."""
        if request.raw_bytes is not None:
            if len(request.raw_bytes) == 0:
                raise ValueError("Uploaded file is empty")
            if len(request.raw_bytes) > self._max_file_bytes:
                raise ValueError(
                    f"File size {len(request.raw_bytes)} exceeds max_file_bytes={self._max_file_bytes}"
                )
        elif not request.content.strip():
            raise ValueError("Either raw_bytes or non-empty content must be provided")

        resolved_title, pages_parsed, chunks = await asyncio.to_thread(
            self._parse_and_chunk,
            request,
        )
        if not chunks:
            raise ValueError("Document yielded no extractable text chunks")

        doc_metadata = dict(request.metadata)
        if pages_parsed > 0:
            doc_metadata["total_pages"] = pages_parsed

        doc = KnowledgeDocumentInput(
            space_id=request.space_id,
            source_type=request.source_type,
            source_id=request.source_id,
            title=resolved_title,
            url=request.url,
            chunks=tuple(chunks),
            metadata=doc_metadata,
        )
        chunks_indexed = await self._knowledge_store.upsert(doc)
        return IngestResult(
            space_id=request.space_id,
            source_type=request.source_type,
            source_id=request.source_id,
            title=resolved_title,
            pages_parsed=pages_parsed,
            chunks_indexed=chunks_indexed,
        )
