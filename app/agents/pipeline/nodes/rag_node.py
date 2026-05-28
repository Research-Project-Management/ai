"""
RAG Node — Document retrieval, reranking, and sufficiency judgment.

Flow:
  1. Translate query to English (for embedding/reranking quality)
  2. Retrieve chunks from Qdrant (filtered by doc_ids + user/chat isolation)
  3. Rerank with configured provider (jina/cohere/local/none)
  4. Format as [D#] context blocks for the LLM
  5. RagJudge: is context sufficient? → route to "done" or "web"
"""

from __future__ import annotations

import asyncio
import json
import logging
import re

from app.agents.pipeline.state import PipelineState
from app.agents.pipeline.reranker import rerank_chunks
from app.config import get_router_llm

logger = logging.getLogger(__name__)

_SUMMARIZE_RE = re.compile(
    r"tóm[\s\-]?tắt|tổng[\s\-]?hợp|tổng[\s\-]?quan|toàn[\s\-]?bộ"
    r"|nội[\s\-]?dung|mô[\s\-]?tả"
    r"|summarize|summary|overview|what.{0,10}(in|about|contain)",
    re.IGNORECASE,
)


def _is_summarize(text: str) -> bool:
    return bool(_SUMMARIZE_RE.search(text))


def _format_rag_context(
    chunks: list[dict],
    max_docs: int = 10,
    max_chars_per_doc: int = 2000,
    max_total_chars: int = 20000,
) -> str:
    """Format retrieved chunks into [D#] blocks for the LLM context.

    Format::
        [D1] file=paper.pdf page=3
        TEXT: The proposed method achieves...
    """
    lines = []
    total = 0
    for i, c in enumerate(chunks[:max_docs], 1):
        source = c.get("source", "unknown")
        page = c.get("metadata", {}).get("page")
        content = (c.get("content", "")).strip()

        if len(content) > max_chars_per_doc:
            content = content[:max_chars_per_doc] + "..."

        header = f"[D{i}] file={source}"
        if page is not None:
            header += f" page={page}"

        block = f"{header}\nTEXT: {content}\n"
        if total + len(block) > max_total_chars:
            break

        lines.append(block)
        total += len(block)

    return "\n".join(lines).strip()


async def rag_node(state: PipelineState) -> dict:
    """Retrieve documents, optionally rerank, and judge sufficiency."""
    logger.info("════ RAG Node ════")

    query_original = state.get("query_original", "")
    document_ids = state.get("document_ids", [])
    web_search_enabled = state.get("web_search_enabled", False)
    user_id = state.get("user_id") or ""
    chat_id = state.get("chat_id") or ""

    if not query_original:
        return {"rag_context": "", "route": "done", "sources": []}

    # ── RAG isolation guard ────────────────────────────────────────────────
    # Without document_ids or chat_id we can only filter by user_id, which
    # would return chunks from *every* previous session — a data leak.
    # Bail out early and fall through to web-search or chat instead.
    if not document_ids and not chat_id:
        logger.info(
            "RAG skipped: no document_ids and no chat_id (user=%s) — "
            "would leak cross-session data",
            user_id,
        )
        return {
            "rag_context": "",
            "route": "web" if web_search_enabled else "done",
            "sources": [],
        }

    metadata = state.get("metadata", {}) or {}
    selected_excerpt = str(metadata.get("selection") or "").strip()
    retrieval_query = query_original
    if selected_excerpt:
        retrieval_query = f"{query_original}\n\nSelected excerpt from the target document:\n{selected_excerpt[:1500]}"

    # Translate for reranking
    from app.agents.utils import translate_to_english_async
    query_en = await translate_to_english_async(retrieval_query)

    # Determine retrieval strategy
    if _is_summarize(query_original):
        rag_query = "main topics key findings methodology results conclusions introduction"
        rag_k = 25
        logger.info("Summarise request → coverage query, k=%d", rag_k)
    else:
        rag_query = query_en
        rag_k = 15
        logger.info("Targeted query: %r, k=%d", rag_query[:60], rag_k)

    # Retrieve
    chunks: list[dict] = []
    try:
        from app.rag.retriever import retrieve_documents

        user_id = state.get("user_id") or ""
        chat_id = state.get("chat_id") or ""

        chunks = await retrieve_documents(
            rag_query,
            k=rag_k,
            document_ids=document_ids or None,
            user_id=user_id or None,
            chat_id=chat_id or None,
        )
        logger.info("Retrieved %d chunks (user=%s chat=%s)", len(chunks), user_id, chat_id)
    except Exception as exc:
        logger.warning("RAG retrieval failed: %s", exc)

    if not chunks:
        logger.info("No chunks → route to web" if web_search_enabled else "No chunks → done")
        return {
            "rag_context": "",
            "query_en": query_en,
            "route": "web" if web_search_enabled else "done",
            "sources": [],
        }

    # Rerank
    chunks = await rerank_chunks(query_en, chunks, min(10, len(chunks)))
    logger.info("After rerank: %d chunks", len(chunks))

    # Format context
    rag_context = _format_rag_context(chunks)

    # Build sources for FE
    seen_sources: dict[str, str] = {}
    for c in chunks:
        src = c.get("source", "")
        if src and src not in seen_sources:
            seen_sources[src] = c.get("content", "")[:400]
    sources = [{"source": src, "snippet": snippet} for src, snippet in seen_sources.items()]

    # ─── RagJudge: is context sufficient? ───
    next_route = "done"

    # Smart skip: if we have enough high-quality chunks, skip the LLM judge call
    avg_score = sum(c.get("score", 0) for c in chunks) / len(chunks) if chunks else 0
    if len(chunks) >= 5 and avg_score >= 0.5:
        logger.info(
            "RagJudge SKIPPED: %d chunks, avg_score=%.2f → sufficient",
            len(chunks), avg_score,
        )
    elif not web_search_enabled:
        # No point judging if web search isn't available as fallback
        logger.info("RagJudge SKIPPED: web_search not enabled")
    else:
        # Borderline case — call LLM judge
        try:
            judge_llm = get_router_llm()
            judge_prompt = (
                "You are a judge evaluating whether retrieved text is sufficient "
                "and relevant to answer the question.\n\n"
                "Criteria:\n"
                "- Directly addresses the main question\n"
                "- Enough detail for a clear answer\n"
                "- Specific and relevant, not vague\n\n"
                'Respond with ONLY JSON: {"sufficient": true/false, "reasoning": "brief"}'
            )
            judge_response = await judge_llm.ainvoke([
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": (
                    f"Question: {query_original}\n\n"
                    f"Retrieved context:\n{rag_context[:8000]}\n\n"
                    "Is this sufficient?"
                )},
            ])

            judge_content = judge_response.content if isinstance(judge_response.content, str) else str(judge_response.content)
            judge_clean = judge_content.strip()
            if judge_clean.startswith("```"):
                judge_clean = judge_clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            verdict = json.loads(judge_clean)
            sufficient = verdict.get("sufficient", True)
            logger.info("RagJudge: sufficient=%s reasoning=%s", sufficient, verdict.get("reasoning", ""))

            if not sufficient:
                next_route = "web"

        except Exception as exc:
            logger.warning("RagJudge failed: %s → treat as sufficient", exc)

    return {
        "rag_context": rag_context,
        "query_en": query_en,
        "route": next_route,
        "sources": sources,
    }
