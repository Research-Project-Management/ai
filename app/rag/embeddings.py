"""Embedding wrapper using Google Generative AI (google-genai SDK).

Uses gemini-embedding-001 (3072-dim) which is available via the standard
v1beta endpoint on Google AI Studio API keys.
"""

from __future__ import annotations

from typing import List

from langchain_core.embeddings import Embeddings

from app.config import get_settings

_embeddings: "GeminiEmbeddings | None" = None


class GeminiEmbeddings(Embeddings):
    """LangChain-compatible embeddings using google-genai SDK."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        import google.genai as genai
        self.model = model
        self._client = genai.Client(api_key=api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            response = self._client.models.embed_content(
                model=self.model,
                contents=text,
            )
            results.append(response.embeddings[0].values)
        return results

    def embed_query(self, text: str) -> List[float]:
        response = self._client.models.embed_content(
            model=self.model,
            contents=text,
        )
        return response.embeddings[0].values


def get_embeddings() -> GeminiEmbeddings:
    """Get or create the embeddings model singleton."""
    global _embeddings
    if _embeddings is None:
        settings = get_settings()
        # Use 'models/' prefix if not already present
        model = settings.embedding_model
        if not model.startswith("models/"):
            model = f"models/{model}"
        _embeddings = GeminiEmbeddings(
            api_key=settings.gemini_api_key,
            model=model,
        )
    return _embeddings

