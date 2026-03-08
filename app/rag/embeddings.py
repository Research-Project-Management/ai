"""Embedding wrapper using Google Generative AI."""

from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings

_embeddings: GoogleGenerativeAIEmbeddings | None = None


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Get or create the embeddings model singleton."""
    global _embeddings
    if _embeddings is None:
        settings = get_settings()
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.gemini_api_key,
        )
    return _embeddings
