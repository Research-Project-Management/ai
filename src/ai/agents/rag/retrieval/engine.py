"""Read-side RetrievalEngine: parallel multi-query RRF merge, rerank, and reverse repack."""

import asyncio
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Final

from ai.agents.rag.retrieval.contracts import (
    RankedPassage,
    Reranker,
    RetrievalBatch,
    RetrievalQuery,
)
from ai.shared.contracts.knowledge import KnowledgeHit, KnowledgeStore
from ai.shared.resilience.tool_runner import execute_tool_safely

_TIMELINE_DIGEST_BOOST: Final[float] = 1.35
_SUPERSEDED_DECAY: Final[float] = 0.65
_ACTIVE_BASE_BOOST: Final[float] = 1.20
_RECENCY_MAX_BONUS: Final[float] = 0.15
_POLICY_SOURCE_BONUS: Final[float] = 0.10


def filter_and_repack(
    ranked_desc: Sequence[RankedPassage],
    *,
    min_score: float,
    top_k: int,
) -> tuple[tuple[RankedPassage, ...], tuple[RankedPassage, ...]]:
    """Filter low-relevance passages and return `(reverse_repacked, descending)` views."""
    filtered = [passage for passage in ranked_desc if passage.rerank_score >= min_score]
    top_desc = tuple(filtered[: max(0, top_k)])
    repacked = tuple(reversed(top_desc))
    return repacked, top_desc


_CURRENT_STATE_CUES: tuple[str, ...] = (
    "hiện tại",
    "mới nhất",
    "đang áp dụng",
    "hiện hành",
    "tính đến",
    "hiện nay",
    "bây giờ",
    "latest",
    "current",
    "currently",
    "active",
    "as of now",
)
_TIMELINE_CUES: tuple[str, ...] = (
    "trình tự",
    "lịch sử",
    "trước khi",
    "so sánh",
    "từ tháng",
    "qua các",
    "tiến hóa",
    "ban đầu",
    "timeline",
    "history",
    "progression",
    "evolution",
)


_SYNTHESIS_TIMELINE_CUES: tuple[str, ...] = (
    *_TIMELINE_CUES,
    "xu hướng",
    "tổng hợp",
    "diễn biến",
    "các mốc",
    "trend",
    "across",
    "quarter",
)


def _compile_cue_pattern(cues: Sequence[str]) -> re.Pattern[str]:
    return re.compile(
        r"\b(?:" + "|".join(re.escape(cue) for cue in cues) + r")\b",
        re.IGNORECASE,
    )


_CURRENT_STATE_CUE_RE: Final[re.Pattern[str]] = _compile_cue_pattern(_CURRENT_STATE_CUES)
_TIMELINE_CUE_RE: Final[re.Pattern[str]] = _compile_cue_pattern(_TIMELINE_CUES)
_SYNTHESIS_TIMELINE_CUE_RE: Final[re.Pattern[str]] = _compile_cue_pattern(_SYNTHESIS_TIMELINE_CUES)


def _is_current_state_query(query: str) -> bool:
    """Return True when query asks for latest active truth rather than full timeline history."""
    has_current = bool(_CURRENT_STATE_CUE_RE.search(query))
    has_timeline = bool(_TIMELINE_CUE_RE.search(query))
    return has_current and not has_timeline


def _is_timeline_query(query: str) -> bool:
    """Return True when query synthesizes trends, sequences, or cross-quarter evolution."""
    return bool(_SYNTHESIS_TIMELINE_CUE_RE.search(query))


def _parse_valid_ts(metadata: Mapping[str, Any]) -> float | None:
    raw = metadata.get("valid_from") or metadata.get("effective_at")
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        text = f"{text}T00:00:00+00:00"
    elif text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _temporal_bounds(items: Sequence[KnowledgeHit | RankedPassage]) -> tuple[float, float]:
    timestamps = [ts for item in items if (ts := _parse_valid_ts(item.metadata)) is not None]
    if not timestamps:
        return 0.0, 0.0
    return min(timestamps), max(timestamps)


def _temporal_multiplier(
    *,
    metadata: Mapping[str, Any],
    source_type: str,
    min_ts: float,
    max_ts: float,
    is_current_query: bool = True,
    is_timeline_query: bool = False,
) -> float:
    if not is_current_query:
        if is_timeline_query and source_type == "timeline_digest":
            return _TIMELINE_DIGEST_BOOST
        return 1.0

    lifecycle = str(metadata.get("lifecycle_status") or "active").lower()
    if lifecycle in {"superseded", "archived"}:
        return _SUPERSEDED_DECAY

    recency_bonus = 0.0
    ts = _parse_valid_ts(metadata)
    if ts is not None and max_ts > min_ts:
        recency_bonus = _RECENCY_MAX_BONUS * ((ts - min_ts) / (max_ts - min_ts))
    policy_bonus = _POLICY_SOURCE_BONUS if source_type == "policy" else 0.0
    return _ACTIVE_BASE_BOOST + recency_bonus + policy_bonus


def _apply_temporal_adjustment(
    items: Sequence[RankedPassage],
    *,
    is_current_query: bool,
    is_timeline_query: bool = False,
) -> list[RankedPassage]:
    if (not is_current_query and not is_timeline_query) or not items:
        return list(items)
    min_ts, max_ts = _temporal_bounds(items)
    adjusted: list[RankedPassage] = [
        item.model_copy(
            update={
                "rerank_score": round(
                    item.rerank_score
                    * _temporal_multiplier(
                        metadata=item.metadata,
                        source_type=item.source_type,
                        min_ts=min_ts,
                        max_ts=max_ts,
                        is_current_query=is_current_query,
                        is_timeline_query=is_timeline_query,
                    ),
                    6,
                )
            }
        )
        for item in items
    ]
    adjusted.sort(key=lambda item: item.rerank_score, reverse=True)
    return adjusted


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


_MIN_CHUNKS_PER_SCOPED_DOC = 2
_MAX_MULTI_DOC_TOP_K = 16  # 16 * 384 ~= 6,144 tokens context budget


def _compute_effective_top_k(
    base_top_k: int,
    scoped_doc_count: int,
    *,
    min_chunks_per_scoped_doc: int = _MIN_CHUNKS_PER_SCOPED_DOC,
    max_multi_doc_top_k: int = _MAX_MULTI_DOC_TOP_K,
) -> int:
    """Scale top_k across multiple scoped documents while bounding context token budget."""
    if scoped_doc_count <= 1:
        return base_top_k
    return min(
        max(base_top_k, scoped_doc_count * min_chunks_per_scoped_doc),
        max_multi_doc_top_k,
    )


def _stratify_passages_by_source(
    passages: Sequence[RankedPassage],
    source_ids: Sequence[str],
    *,
    min_score: float,
    top_k: int,
) -> list[RankedPassage]:
    """Guarantee at least one eligible passage per represented source, then fill by score."""
    eligible = [passage for passage in passages if passage.rerank_score >= min_score]
    if len(source_ids) <= 1 or top_k <= 0:
        return eligible[: max(0, top_k)]

    scoped = set(source_ids)
    selected: list[RankedPassage] = []
    selected_ids: set[int] = set()
    represented_sources: set[str] = set()

    for passage in eligible:
        if passage.source_id in scoped and passage.source_id not in represented_sources:
            represented_sources.add(passage.source_id)
            selected.append(passage)
            selected_ids.add(id(passage))
            if len(selected) >= top_k:
                break

    if len(selected) < top_k:
        for passage in eligible:
            if id(passage) in selected_ids:
                continue
            selected.append(passage)
            selected_ids.add(id(passage))
            if len(selected) >= top_k:
                break

    selected.sort(key=lambda item: item.rerank_score, reverse=True)
    return selected


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
        min_chunks_per_scoped_doc: int = _MIN_CHUNKS_PER_SCOPED_DOC,
        max_multi_doc_top_k: int = _MAX_MULTI_DOC_TOP_K,
    ) -> None:
        self._knowledge_store = knowledge_store
        self._reranker = reranker
        self._candidate_pool_size = candidate_pool_size
        self._default_top_k = default_top_k
        self._default_min_score = default_min_score
        self._min_chunks_per_scoped_doc = max(1, min_chunks_per_scoped_doc)
        self._max_multi_doc_top_k = max(2, max_multi_doc_top_k)

    async def retrieve(self, query: RetrievalQuery) -> RetrievalBatch:
        """Execute parallel scoped queries, fuse via RRF, rerank, and reverse-repack."""
        queries = _deduplicate_queries(query.primary_query, query.expanded_queries)
        if not queries:
            return RetrievalBatch(passages=(), ranked_desc=(), total_candidates=0)

        distinct_sources = tuple(dict.fromkeys(s.strip() for s in query.source_ids if s.strip()))
        base_top_k = query.top_k or self._default_top_k
        top_k = _compute_effective_top_k(
            base_top_k,
            len(distinct_sources),
            min_chunks_per_scoped_doc=self._min_chunks_per_scoped_doc,
            max_multi_doc_top_k=self._max_multi_doc_top_k,
        )
        pool_limit = query.candidate_pool_size or self._candidate_pool_size
        if len(distinct_sources) > 1:
            pool_limit = max(pool_limit, top_k * 2)
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

        combined_query = " ".join((query.primary_query, *query.expanded_queries))
        is_current_query = _is_current_state_query(combined_query)
        is_timeline_query = (not is_current_query) and _is_timeline_query(combined_query)
        rerank_k = (
            min(len(fused_candidates), max(top_k * 2, 12))
            if (is_current_query or is_timeline_query or len(distinct_sources) > 1)
            else min(top_k, len(fused_candidates))
        )
        reranked = await self._reranker.rerank(
            query.primary_query,
            fused_candidates[:pool_limit],
            top_k=rerank_k,
        )
        adjusted_reranked = _apply_temporal_adjustment(
            reranked,
            is_current_query=is_current_query,
            is_timeline_query=is_timeline_query,
        )
        final_candidates: Sequence[RankedPassage]
        if len(distinct_sources) > 1:
            final_candidates = _stratify_passages_by_source(
                adjusted_reranked,
                distinct_sources,
                min_score=min_score,
                top_k=top_k,
            )
        else:
            final_candidates = adjusted_reranked
        repacked, ranked_desc = filter_and_repack(
            final_candidates,
            min_score=min_score,
            top_k=top_k,
        )
        return RetrievalBatch(
            passages=repacked,
            ranked_desc=ranked_desc,
            total_candidates=len(fused_candidates),
        )
