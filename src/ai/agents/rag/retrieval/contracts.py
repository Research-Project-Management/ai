"""Contracts for multi-query retrieval, cross-encoder reranking, and reverse repacking."""

import re
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ai.orchestrator.contracts import CitationItem
from ai.shared.contracts.knowledge import KnowledgeHit

_TRAILING_OPEN_TAG_RE = re.compile(r"<[a-zA-Z/][^>]*$")
_BR_TAG_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_INLINE_HTML_RE = re.compile(r"</?(?:b|i|em|strong|sup|sub|span|p|div|code)\b[^>]*>", re.IGNORECASE)
_TABLE_SEP_RE = re.compile(r"^\s*\|?(?:\s*:?-{2,}:?\s*\|)+\s*(?::?-{2,}:?\s*\|?)?\s*$")
_COL_PLACEHOLDER_RE = re.compile(r"^col\d+$", re.IGNORECASE)
_HEADING_PREFIX_RE = re.compile(r"^#{1,6}\s+")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def format_citation_snippet(raw_text: str, *, max_chars: int = 280) -> str:
    """Normalize table/HTML artifacts into readable excerpt text and truncate cleanly."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    text = _TRAILING_OPEN_TAG_RE.sub("", text)
    text = _BR_TAG_RE.sub("; ", text)
    text = _INLINE_HTML_RE.sub("", text)

    cleaned_lines: list[str] = []
    for raw_line in text.split("\n"):
        stripped = raw_line.strip()
        if not stripped or _TABLE_SEP_RE.match(stripped):
            continue

        if "|" in stripped and (stripped.startswith("|") or stripped.endswith("|")):
            cells: list[str] = []
            for cell in stripped.split("|"):
                normalized_cell = _MULTI_SPACE_RE.sub(" ", cell.strip())
                if not normalized_cell or _COL_PLACEHOLDER_RE.match(normalized_cell):
                    continue
                if cells and cells[-1] == normalized_cell:
                    continue
                cells.append(normalized_cell)
            if cells:
                cleaned_lines.append(" · ".join(cells))
            continue

        no_heading = _HEADING_PREFIX_RE.sub("", stripped)
        normalized_line = _MULTI_SPACE_RE.sub(" ", no_heading).strip()
        if normalized_line:
            cleaned_lines.append(normalized_line)

    cleaned = "\n".join(cleaned_lines).strip()
    if max_chars > 0 and len(cleaned) > max_chars:
        budget = max(1, max_chars - 1)
        window = cleaned[:budget]
        cut = max(window.rfind(" "), window.rfind("\n"))
        if cut >= len(window) // 2:
            window = window[:cut]
        cleaned = window.rstrip(".,;:|· \n\t") + "…"
    return cleaned


class RankedPassage(BaseModel):
    """Immutable evidence passage scored by both hybrid retrieval and reranking."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    source_type: str
    title: str = ""
    content: str
    rerank_score: float
    retrieval_score: float
    chunk_index: int = 0
    page: int | None = None
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_citation(self, *, max_snippet_chars: int = 280) -> CitationItem:
        """Convert passage into a grounded CitationItem using unprefixed snippet text."""
        raw_text = str(self.metadata.get("raw_content") or self.content).strip()
        return CitationItem(
            source_id=self.source_id,
            title=self.title,
            url=self.url,
            page=self.page,
            snippet=format_citation_snippet(raw_text, max_chars=max_snippet_chars),
        )

    def to_knowledge_hit(self) -> KnowledgeHit:
        """Project ranked passage back to KnowledgeHit schema for /knowledge/search."""
        return KnowledgeHit(
            source_id=self.source_id,
            source_type=self.source_type,
            title=self.title,
            content=self.content,
            score=self.rerank_score,
            chunk_index=self.chunk_index,
            page=self.page,
            url=self.url,
            metadata=self.metadata,
        )


@runtime_checkable
class Reranker(Protocol):
    """Port interface for scoring candidate passages against a user query."""

    async def rerank(
        self,
        query: str,
        candidates: Sequence[KnowledgeHit],
        *,
        top_k: int,
    ) -> list[RankedPassage]:
        """Return candidate passages sorted in descending order of relevance."""
        ...


class RetrievalQuery(BaseModel):
    """Immutable query specification accepted by RetrievalEngine."""

    model_config = ConfigDict(frozen=True)

    space_id: str = Field(..., min_length=1)
    primary_query: str = Field(..., min_length=1)
    expanded_queries: tuple[str, ...] = ()
    source_types: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    candidate_pool_size: int | None = None
    top_k: int | None = None
    min_score: float | None = None


class RetrievalBatch(BaseModel):
    """Immutable retrieval output offering both reverse-repacked and descending views."""

    model_config = ConfigDict(frozen=True)

    passages: tuple[RankedPassage, ...]
    ranked_desc: tuple[RankedPassage, ...]
    total_candidates: int = 0
