"""Reranker adapters with remote-to-local fallback."""

import itertools
import logging
import re
from collections.abc import Sequence
from typing import Final

import httpx

from ai.agents.rag.retrieval.contracts import RankedPassage, Reranker
from ai.shared.config.rag import RerankConfig
from ai.shared.contracts.knowledge import KnowledgeHit

logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"\w+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "into",
        "is",
        "it",
        "much",
        "of",
        "on",
        "or",
        "summarize",
        "summary",
        "explain",
        "describe",
        "tell",
        "that",
        "the",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "là",
        "và",
        "của",
        "trong",
        "cho",
        "các",
        "những",
        "một",
        "như",
        "thế",
        "nào",
        "tóm",
        "tắt",
    }
)
_REF_QUERY_RE = re.compile(
    r"\b(references?|bibliography|citations?|cited\s+works?)\b|tài\s+liệu\s+tham\s+khảo|trích\s+dẫn",
    re.IGNORECASE,
)
_REF_SECTION_RE = re.compile(
    r"(?:^|>|\s)(?:references?|bibliography|works\s+cited)\s*$",
    re.IGNORECASE,
)
_REF_ENTRY_RE = re.compile(r"(?:^|\n)\s*\[\d+\]\s+[A-Z]")

_RANK_PRIOR_DECAY: Final[float] = 0.20
_PRIOR_WEIGHT_WITH_MATCH: Final[float] = 0.22
_PRIOR_WEIGHT_NO_MATCH: Final[float] = 0.03
_TOKEN_RECALL_WEIGHT: Final[float] = 0.48
_BIGRAM_SCORE_WEIGHT: Final[float] = 0.14
_SECTION_OVERLAP_WEIGHT: Final[float] = 0.12
_TITLE_OVERLAP_WEIGHT: Final[float] = 0.04
_BIBLIOGRAPHY_PENALTY_FACTOR: Final[float] = 0.45


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _content_tokens(tokens: Sequence[str]) -> set[str]:
    filtered = {t for t in tokens if t not in _STOPWORDS and len(t) > 1}
    return filtered if filtered else set(tokens)


def _is_bibliography_chunk(hit: KnowledgeHit) -> bool:
    section = str(hit.metadata.get("section") or "")
    return bool(_REF_SECTION_RE.search(section)) or len(_REF_ENTRY_RE.findall(hit.content)) >= 3


class LexicalHybridReranker:
    """Fast local zero-network reranker blending lexical content recall, section match, and RRF prior."""

    async def rerank(
        self,
        query: str,
        candidates: Sequence[KnowledgeHit],
        *,
        top_k: int,
    ) -> list[RankedPassage]:
        if not candidates or top_k <= 0:
            return []

        wants_refs = bool(_REF_QUERY_RE.search(query))
        q_tokens = _tokenize(query)
        q_set = _content_tokens(q_tokens)
        q_bigrams = set(itertools.pairwise(q_tokens))

        scored: list[RankedPassage] = []
        for idx, hit in enumerate(candidates):
            rank_prior = 1.0 / (1.0 + idx * _RANK_PRIOR_DECAY)
            body_tokens = _tokenize(hit.content)
            body_set = set(body_tokens)
            section_set = _content_tokens(_tokenize(str(hit.metadata.get("section", ""))))
            title_set = _content_tokens(_tokenize(hit.title))

            if q_set:
                token_recall = len(q_set & body_set) / len(q_set)
                section_overlap = len(q_set & section_set) / len(q_set)
                title_overlap = len(q_set & title_set) / len(q_set)
            else:
                token_recall = 0.0
                section_overlap = 0.0
                title_overlap = 0.0

            if q_bigrams:
                body_bigrams = set(itertools.pairwise(body_tokens))
                bigram_score = len(q_bigrams & body_bigrams) / len(q_bigrams)
            else:
                bigram_score = 0.0

            has_lexical_match = (
                token_recall > 0.0
                or bigram_score > 0.0
                or section_overlap > 0.0
                or title_overlap > 0.0
            )
            prior_weight = (
                _PRIOR_WEIGHT_WITH_MATCH
                if (has_lexical_match or not q_set)
                else _PRIOR_WEIGHT_NO_MATCH
            )
            combined = (
                prior_weight * rank_prior
                + _TOKEN_RECALL_WEIGHT * token_recall
                + _BIGRAM_SCORE_WEIGHT * bigram_score
                + _SECTION_OVERLAP_WEIGHT * section_overlap
                + _TITLE_OVERLAP_WEIGHT * title_overlap
            )
            if not wants_refs and _is_bibliography_chunk(hit):
                combined *= _BIBLIOGRAPHY_PENALTY_FACTOR
            scored.append(
                RankedPassage(
                    source_id=hit.source_id,
                    source_type=hit.source_type,
                    title=hit.title,
                    content=hit.content,
                    rerank_score=round(combined, 6),
                    retrieval_score=hit.score,
                    chunk_index=hit.chunk_index,
                    page=hit.page,
                    url=hit.url,
                    metadata=hit.metadata,
                )
            )

        scored.sort(key=lambda item: item.rerank_score, reverse=True)
        return scored[:top_k]


class HttpCrossEncoderReranker:
    """Cross-encoder HTTP reranker (Cohere/Jina/TEI/vLLM compatible) with local fallback."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        model: str = "bge-reranker-v2-m3",
        timeout_seconds: float = 8.0,
        fallback: Reranker | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._endpoint = f"{base_url.rstrip('/')}/rerank"
        self._api_key = api_key.strip()
        self._model = model
        self._timeout = timeout_seconds
        self._fallback: Reranker = fallback or LexicalHybridReranker()
        self._client = http_client

    async def rerank(
        self,
        query: str,
        candidates: Sequence[KnowledgeHit],
        *,
        top_k: int,
    ) -> list[RankedPassage]:
        if not candidates or top_k <= 0:
            return []

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "query": query,
            "documents": [hit.content for hit in candidates],
            "top_n": min(top_k, len(candidates)),
        }

        try:
            if self._client is not None:
                response = await self._client.post(self._endpoint, json=payload, headers=headers)
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(self._endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            raw_results = data.get("results") or []
            if not isinstance(raw_results, list) or not raw_results:
                return await self._fallback.rerank(query, candidates, top_k=top_k)

            ranked: list[RankedPassage] = []
            seen_indices: set[int] = set()
            for entry in raw_results:
                if not isinstance(entry, dict):
                    continue
                idx = int(entry["index"])
                if idx < 0 or idx >= len(candidates) or idx in seen_indices:
                    continue
                seen_indices.add(idx)
                hit = candidates[idx]
                score = float(entry.get("relevance_score", entry.get("score", 0.0)))
                ranked.append(
                    RankedPassage(
                        source_id=hit.source_id,
                        source_type=hit.source_type,
                        title=hit.title,
                        content=hit.content,
                        rerank_score=score,
                        retrieval_score=hit.score,
                        chunk_index=hit.chunk_index,
                        page=hit.page,
                        url=hit.url,
                        metadata=hit.metadata,
                    )
                )

            if not ranked:
                return await self._fallback.rerank(query, candidates, top_k=top_k)

            ranked.sort(key=lambda item: item.rerank_score, reverse=True)
            return ranked[:top_k]
        except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError) as exc:
            logger.warning(
                "Remote reranker failed (%s); falling back to LexicalHybridReranker",
                type(exc).__name__,
            )
            return await self._fallback.rerank(query, candidates, top_k=top_k)


def build_reranker(
    config: RerankConfig,
    *,
    allow_remote: bool = True,
    http_client: httpx.AsyncClient | None = None,
) -> Reranker:
    """Build remote HTTP cross-encoder reranker or local lexical hybrid reranker."""
    local = LexicalHybridReranker()
    if not allow_remote or not config.is_remote_configured:
        return local
    return HttpCrossEncoderReranker(
        base_url=config.base_url,
        api_key=config.api_key.get_secret_value(),
        model=config.model,
        timeout_seconds=config.timeout_seconds,
        fallback=local,
        http_client=http_client,
    )
