"""Contracts for vector embeddings and space-isolated hybrid knowledge storage."""

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


@runtime_checkable
class Embedder(Protocol):
    """Port interface for text embedding providers."""

    @property
    def dimensions(self) -> int:
        """Vector dimensionality produced by this embedder."""
        ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of document chunks into dense vectors."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a search query into a dense vector."""
        ...


class KnowledgeChunkInput(BaseModel):
    """Explicit chunk payload supplied for indexing."""

    model_config = ConfigDict(frozen=True)

    chunk_index: int = Field(default=0, ge=0)
    content: str = Field(..., min_length=1)
    page: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeDocumentInput(BaseModel):
    """Canonical source document payload indexed into a Flux space."""

    model_config = ConfigDict(frozen=True)

    space_id: str = Field(..., min_length=1)
    source_type: str = Field(..., min_length=1)
    source_id: str = Field(..., min_length=1)
    title: str = ""
    url: str | None = None
    content: str = ""
    chunks: tuple[KnowledgeChunkInput, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeHit(BaseModel):
    """Ranked hybrid search match scoped strictly to a Flux space."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    source_type: str
    title: str = ""
    content: str
    score: float
    chunk_index: int = 0
    page: int | None = None
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class KnowledgeStore(Protocol):
    """Port interface for idempotent vector indexing and scoped retrieval."""

    async def upsert(self, doc: KnowledgeDocumentInput) -> int:
        """Index or replace all chunks for a source inside its `space_id`."""
        ...

    async def has_source(self, *, space_id: str, source_id: str) -> bool:
        """Return True when at least one chunk exists for `source_id` in `space_id`."""
        ...

    async def search(
        self,
        *,
        space_id: str,
        query: str,
        limit: int = 5,
        source_types: Sequence[str] = (),
        source_ids: Sequence[str] = (),
    ) -> list[KnowledgeHit]:
        """Search indexed chunks isolated strictly to `space_id`."""
        ...

    async def delete_source(self, *, space_id: str, source_type: str, source_id: str) -> bool:
        """Delete all chunks belonging to one source in `space_id`."""
        ...

    async def delete_space(self, *, space_id: str) -> bool:
        """Delete all indexed knowledge chunks belonging to `space_id`."""
        ...
