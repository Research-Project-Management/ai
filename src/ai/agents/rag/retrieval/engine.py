"""Read-side RetrievalEngine: parallel multi-query RRF merge, rerank, and reverse repack."""

import asyncio
from collections.abc import Sequence

from ai.agents.rag.retrieval.contracts import Reranker, RetrievalBatch, RetrievalQuery
from ai.agents.rag.retrieval.repacker import filter_and_repack
from ai.shared.contracts.knowledge import KnowledgeHit, KnowledgeStore
from ai.shared.resilience.tool_runner import execute_tool_safely


def _deduplicate_queries(primary: str, expanded: Sequence[str], *, limit: int = 4) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in (primary, *expanded):
        cleaned = candidate.strip()
        lowered = cleaned.lower()
        if cleaned and lowered not in seen:
            seen.add(lowered)
            ordered.append(cleaned)
            if len(ordered) >= limit:
                break
    return ordered


def _fuse_rrf(result_lists: Sequence[list[KnowledgeHit]], *, k: int = 60) -> list[KnowledgeHit]:
    scores: dict[tuple[str, str, int], float] = {}
    best_hit: dict[tuple[str, str, int], KnowledgeHit] = {}

    for hits in result_lists:
        for rank, hit in enumerate(hits, start=1):
            key = (hit.source_type, hit.source_id, hit.chunk_index)
            scores[key] = scores.get(key, 0.0) + (1.0 / (k + rank))
            if key not in best_hit or hit.score > best_hit[key].score:
                best_hit[key] = hit

    merged: list[KnowledgeHit] = []
    for key, rrf_score in sorted(scores.items(), key=lambda pair: pair[1], reverse=True):
        merged.append(best_hit[key].model_copy(update={"score": round(rrf_score, 6)}))
    return merged


class RetrievalEngine:
    """Deep IR engine reusable by both `/knowledge/search` and Specialist Agents."""

    def __init__(
        self,
        *,
        knowledge_store: KnowledgeStore,
        reranker: Reranker,
        candidate_pool_size: int = 24,
        default_top_k: int = 6,
        default_min_score: float = 0.05,
    ) -> None:
        self._knowledge_store = knowledge_store
        self._reranker = reranker
        self._candidate_pool_size = candidate_pool_size
        self._default_top_k = default_top_k
        self._default_min_score = default_min_score

    async def retrieve(self, query: RetrievalQuery) -> RetrievalBatch:
        """Execute parallel scoped queries, fuse via RRF, rerank, and reverse-repack."""
        queries = _deduplicate_queries(query.primary_query, query.expanded_queries)
        if not queries:
            return RetrievalBatch(passages=(), ranked_desc=(), total_candidates=0)

        pool_limit = query.candidate_pool_size or self._candidate_pool_size
        top_k = query.top_k or self._default_top_k
        min_score = self._default_min_score if query.min_score is None else query.min_score

        search_coros = [
            execute_tool_safely(
                self._knowledge_store.search,
                space_id=query.space_id,
                query=q_text,
                limit=pool_limit,
                source_types=query.source_types,
                source_ids=query.source_ids,
            )
            for q_text in queries
        ]
        raw_batches = await asyncio.gather(*search_coros)
        fused_candidates = _fuse_rrf(raw_batches)
        if not fused_candidates:
            return RetrievalBatch(passages=(), ranked_desc=(), total_candidates=0)

        reranked = await self._reranker.rerank(
            query.primary_query,
            fused_candidates[:pool_limit],
            top_k=min(top_k, len(fused_candidates)),
        )
        repacked, ranked_desc = filter_and_repack(
            reranked,
            min_score=min_score,
            top_k=top_k,
        )
        return RetrievalBatch(
            passages=repacked,
            ranked_desc=ranked_desc,
            total_candidates=len(fused_candidates),
        )
