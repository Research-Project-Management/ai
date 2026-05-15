"""
Reranker — multi-provider chunk reranking for RAG retrieval.

Provider priority (set RERANKER_PROVIDER in .env):
  "none"  → skip reranking, use Qdrant similarity order (default)
  "jina"  → Jina Reranker API (~80ms, no local model, 1M tokens/month free)
  "cohere"→ Cohere Rerank API (100 reranks/min free tier)
  "local" → CrossEncoder local model (requires ~2.27GB download)
"""

from __future__ import annotations

import logging

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Local CrossEncoder state (lazy-loaded) ────────────────────────────────

_reranker = None
_reranker_loaded = False


# ── Provider implementations ──────────────────────────────────────────────

async def _rerank_jina(query: str, chunks: list[dict], top_k: int, api_key: str) -> list[dict]:
    """Rerank using Jina Reranker API."""
    import httpx

    documents = [(c.get("content", ""))[:2000] for c in chunks]
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://api.jina.ai/v1/rerank",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": documents,
                "top_n": top_k,
            },
        )
        if resp.status_code != 200:
            logger.warning("Jina reranker error %d — using original order", resp.status_code)
            return chunks[:top_k]
        results = resp.json().get("results", [])
        ranked = sorted(results, key=lambda r: r.get("relevance_score", 0), reverse=True)
        return [chunks[r["index"]] for r in ranked if r["index"] < len(chunks)]


async def _rerank_cohere(query: str, chunks: list[dict], top_k: int, api_key: str) -> list[dict]:
    """Rerank using Cohere Rerank API."""
    import httpx

    documents = [(c.get("content", ""))[:2000] for c in chunks]
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://api.cohere.com/v2/rerank",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "rerank-v3.5",
                "query": query,
                "documents": documents,
                "top_n": top_k,
            },
        )
        if resp.status_code != 200:
            logger.warning("Cohere reranker error %d — using original order", resp.status_code)
            return chunks[:top_k]
        results = resp.json().get("results", [])
        ranked = sorted(results, key=lambda r: r.get("relevance_score", 0), reverse=True)
        return [chunks[r["index"]] for r in ranked if r["index"] < len(chunks)]


def _rerank_local(query: str, chunks: list[dict], top_k: int, model_name: str) -> list[dict]:
    """Rerank using local CrossEncoder (requires ~2.27GB model download)."""
    global _reranker, _reranker_loaded
    if not _reranker_loaded:
        _reranker_loaded = True
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(model_name)
            logger.info("CrossEncoder loaded: %s", model_name)
        except Exception as exc:
            logger.warning("CrossEncoder load failed: %s", exc)

    if not _reranker:
        return chunks[:top_k]

    pairs = [[query, (c.get("content", ""))[:2000]] for c in chunks]
    try:
        scores = _reranker.predict(pairs)
        scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]
    except Exception as exc:
        logger.warning("CrossEncoder predict failed: %s", exc)
        return chunks[:top_k]


# ── Public API ────────────────────────────────────────────────────────────

async def rerank_chunks(query: str, chunks: list[dict], top_k: int = 10) -> list[dict]:
    """Rerank retrieved chunks using the configured provider.

    Falls back to original Qdrant similarity order on any error.
    """
    import asyncio

    if not chunks:
        return chunks

    settings = get_settings()
    provider = settings.reranker_provider.lower()

    if provider in ("none", ""):
        return chunks[:top_k]

    if provider == "jina":
        if not settings.jina_api_key:
            logger.warning("JINA_API_KEY not set — skipping reranking")
            return chunks[:top_k]
        try:
            return await _rerank_jina(query, chunks, top_k, settings.jina_api_key)
        except Exception as exc:
            logger.warning("Jina reranker failed: %s — using original order", exc)
            return chunks[:top_k]

    if provider == "cohere":
        if not settings.cohere_api_key:
            logger.warning("COHERE_API_KEY not set — skipping reranking")
            return chunks[:top_k]
        try:
            return await _rerank_cohere(query, chunks, top_k, settings.cohere_api_key)
        except Exception as exc:
            logger.warning("Cohere reranker failed: %s — using original order", exc)
            return chunks[:top_k]

    if provider == "local":
        return await asyncio.to_thread(
            _rerank_local, query, chunks, top_k, settings.cross_encoder_model
        )

    logger.warning("Unknown reranker_provider '%s' — skipping reranking", provider)
    return chunks[:top_k]
