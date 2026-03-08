"""Tests for the RAG pipeline."""

from app.rag.ingestion import splitter


class TestTextSplitter:
    """Test the document chunking logic."""

    def test_split_short_text(self):
        """Short text should produce one chunk."""
        text = "This is a short text."
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_long_text(self):
        """Long text should be split into multiple chunks."""
        text = "Hello world. " * 200  # ~2600 chars
        chunks = splitter.split_text(text)
        assert len(chunks) > 1

    def test_chunks_have_overlap(self):
        """Adjacent chunks should have overlapping content."""
        text = " ".join(f"sentence-{i}" for i in range(200))
        chunks = splitter.split_text(text)
        if len(chunks) > 1:
            # Check some overlap exists between chunk 0 and chunk 1
            words_0 = set(chunks[0].split())
            words_1 = set(chunks[1].split())
            overlap = words_0 & words_1
            assert len(overlap) > 0, "Chunks should have overlapping content"
