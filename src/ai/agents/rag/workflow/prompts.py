"""Isolated system prompts and evidence formatting for `rag_agent`."""

from collections.abc import Sequence

from ai.agents.rag.retrieval.contracts import RankedPassage
from ai.orchestrator.citations import CITATION_INSTRUCTIONS, format_evidence_tag


def build_query_transform_prompt(
    *,
    attempt: int,
    max_queries: int,
    prior_queries: Sequence[str] = (),
) -> str:
    """Build prompt instructing the LLM to rewrite and expand search queries."""
    retry_note = ""
    if attempt > 1 and prior_queries:
        prior_str = ", ".join(repr(q) for q in prior_queries)
        retry_note = (
            f"\nPrevious search queries returned insufficient evidence: {prior_str}. "
            "Generate broader synonyms, alternative technical phrasings, and cross-lingual keywords."
        )

    return (
        "You are a precision query planner for hybrid document retrieval (Dense + BM25).\n"
        "Rewrite the user request into a standalone query (resolving pronouns from conversation tail) "
        f"and produce 1 to {max_queries} focused search queries.\n"
        "Include both the user's language and concise English technical terms if the request is non-English."
        f"{retry_note}\n\n"
        "Return strict JSON only (do NOT invoke any native tool/function calls, and do NOT emit `<think>` blocks):\n"
        '{"standalone_query": "...", "search_queries": ["query 1", "query 2"]}'
    )


def build_grounded_answer_prompt() -> str:
    """Build ALCE-style grounded citation synthesis system prompt."""
    return (
        "You are a rigorous document-reading assistant.\n"
        "Answer ONLY using facts directly supported by the numbered evidence passages below.\n"
        f"{CITATION_INSTRUCTIONS}\n"
        "Never invent facts or metrics not present in the evidence.\n"
        "If the evidence only partially answers the question, state clearly what is supported "
        "and what is missing. Reply in the same language as the user's question."
    )


def format_evidence_block(
    *,
    query: str,
    passages_repacked: Sequence[RankedPassage],
    ranked_desc: Sequence[RankedPassage],
) -> str:
    """Format passages in reverse-repacked order while preserving canonical [cite:K] citation IDs."""
    index_by_key = {
        (p.source_type, p.source_id, p.chunk_index): idx
        for idx, p in enumerate(ranked_desc, start=1)
    }
    blocks: list[str] = []
    for passage in passages_repacked:
        idx = index_by_key[(passage.source_type, passage.source_id, passage.chunk_index)]
        raw_total = passage.metadata.get("total_pages")
        total_pages = (
            raw_total
            if isinstance(raw_total, int) and not isinstance(raw_total, bool) and raw_total > 0
            else None
        )
        if passage.page is not None and total_pages is not None:
            page_label = f" | Page {passage.page}/{total_pages}"
        elif passage.page is not None:
            page_label = f" | Page {passage.page}"
        elif total_pages is not None:
            page_label = f" | Total Pages: {total_pages}"
        else:
            page_label = ""
        title_label = passage.title or passage.source_id
        blocks.append(
            f"{format_evidence_tag(idx)} Source: {title_label}{page_label}\n{passage.content.strip()}"
        )

    joined = "\n\n---\n\n".join(blocks)
    return f"Evidence (ordered with strongest match closest to the question):\n\n{joined}\n\nQuestion: {query}"
