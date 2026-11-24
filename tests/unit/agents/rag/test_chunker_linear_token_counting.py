"""Seam tests verifying linear O(N) token counting in MarkdownStructureChunker."""

from __future__ import annotations

import pytest

from ai.agents.rag.ingestion.chunker import MarkdownStructureChunker
from ai.agents.rag.ingestion.contracts import ParsedDocument, ParsedPage


def test_chunker_counts_tokens_linearly_without_reencoding_accumulated_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunker = MarkdownStructureChunker(chunk_size_tokens=2000, chunk_overlap_tokens=32)

    paragraphs = [
        f"Paragraph {i}: attention scaling improves latency by {i} percent." for i in range(60)
    ]
    markdown = "# Scaling Study\n\n" + "\n\n".join(paragraphs)
    doc = ParsedDocument(
        title="Scaling Paper",
        pages=(ParsedPage(page_number=1, markdown=markdown),),
    )

    encoded_chars = 0
    original_count = chunker.count_tokens

    def _counting_wrapper(text: str) -> int:
        nonlocal encoded_chars
        encoded_chars += len(text)
        return original_count(text)

    monkeypatch.setattr(chunker, "count_tokens", _counting_wrapper)

    chunks = chunker.chunk(doc)
    assert len(chunks) == 1
    assert "Paragraph 0:" in chunks[0].content
    assert "Paragraph 59:" in chunks[0].content
    # O(N) per-block counting encodes each block once (<= 1.2x markdown length),
    # whereas O(N^2) re-encoding the growing buffer encodes ~30x markdown length.
    assert encoded_chars <= int(len(markdown) * 1.25)


def test_chunker_preserves_heading_overlap_and_table_boundaries_with_tracked_counts() -> None:
    chunker = MarkdownStructureChunker(chunk_size_tokens=40, chunk_overlap_tokens=8)
    markdown = (
        "## Results\n\n"
        "First finding about sparse attention reducing KV cache memory footprint significantly.\n\n"
        "Second finding about quantization preserving downstream reasoning accuracy across tasks.\n\n"
        "| Model | Score |\n"
        "| --- | --- |\n"
        "| Flux-A | 91.2 |\n"
        "| Flux-B | 94.5 |"
    )
    doc = ParsedDocument(
        title="Eval",
        pages=(ParsedPage(page_number=2, markdown=markdown),),
    )

    chunks = chunker.chunk(doc)
    assert len(chunks) >= 2
    assert all(c.page == 2 for c in chunks)
    assert any("| Flux-B | 94.5 |" in c.content for c in chunks)
    assert all(c.metadata.get("section") == "Results" for c in chunks)
