"""Threshold filtering and Reverse Repacking to mitigate Lost-in-the-Middle."""

from collections.abc import Sequence

from ai.agents.rag.retrieval.contracts import RankedPassage


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
