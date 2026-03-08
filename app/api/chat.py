"""
Chat endpoint — unified gateway to the multi-agent system.

New pipeline (streaming path):
  1. Classify: is this a specialized request (latex / analyze / task)?
     If yes → stream with that agent's system prompt directly.
  2. For general requests:
     a. Rewrite the user query to a concise search-optimized form.
     b. Always try RAG (score >= RAG_SCORE_THRESHOLD, or score >= 0 if doc_ids pinned).
     c. RAG returned good chunks  → intent = "rag".
     d. Else, if web_search_sites enabled → run WebSearchAgent.fetch_results_with_query().
           Results found → intent = "web_search".
           No results    → intent = "chat"  (inform user).
     e. Otherwise → intent = "chat" (plain LLM answer).
  3. Send early META {agent, intent}.
  4. Stream tokens.
  5. Send final META {agent, intent, sources}.

Sources are included in step 5 so the FE can persist them with each message.
"""

from __future__ import annotations

import json
import logging
import re

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.registry import get_agent
from app.agents.router import classify_intent
from app.config import get_llm
from app.models import AgentOutput, ChatRequest, ChatResponse
from app.rag.retriever import retrieve_documents

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# Agents that skip the RAG/web-search pipeline and go directly to the LLM.
# "latex" is kept here intentionally — it generates code and doesn’t need docs.
SPECIALIZED_AGENTS = {"latex", "analyze", "task"}

# Summarisation detection: match common Vietnamese + English keywords.
# Applied to the RAW user query (before any LLM rewriting) so Vietnamese
# terms like “tóm tắt” are always caught.
_SUMMARIZE_RE = re.compile(
    r"tóm[\s\-]?tắt|tổng[\s\-]?hợp|tổng[\s\-]?quan|toàn[\s\-]?bộ"
    r"|nội[\s\-]?dung|mô[\s\-]?tả"
    r"|summarize|summary|overview|what.{0,10}(in|about|contain)",
    re.IGNORECASE,
)


def _is_summarize(text: str) -> bool:
    return bool(_SUMMARIZE_RE.search(text))



# ── Helpers ───────────────────────────────────────────────────────────────────

async def _rewrite_query(query: str) -> str:
    """Rewrite a verbose user message into a short, search-optimized query.

    For short queries (<= 30 chars) the original is returned as-is to save latency.
    """
    if len(query) <= 30:
        return query
    try:
        llm = get_llm(temperature=0)
        response = await llm.ainvoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a search query optimizer. "
                        "Rewrite the user's message as a short, keyword-focused search query "
                        "(max 10 words). Return ONLY the rewritten query — no explanation, no quotes."
                    ),
                },
                {"role": "user", "content": query},
            ]
        )
        rewritten = (
            response.content.strip() if isinstance(response.content, str) else query
        )
        logger.info("Query rewritten: %r → %r", query, rewritten)
        return rewritten or query
    except Exception as exc:
        logger.warning("Query rewrite failed, using original: %s", exc)
        return query


def _build_system_prompt(
    intent: str,
    rag_chunks: list[dict],
    web_results: list[dict],
    metadata: dict,
) -> str:
    """Return the appropriate system prompt string for the resolved intent."""

    if intent == "rag":
        ctx_text = (
            "\n\n---\n\n".join(
                f"[Source: {c.get('source', '?')}]\n{c.get('content', '')}"
                for c in rag_chunks
            )
            if rag_chunks
            else "(No content could be retrieved from the selected sources.)"
        )
        return (
            "You are a research assistant. Your task is to answer questions STRICTLY "
            "based on the user's uploaded source documents provided below.\n"
            "Rules:\n"
            "1. Only use information that appears in the Source Passages.\n"
            "2. Cite each fact inline as **[Source: filename]**.\n"
            "3. For summarization requests, produce a well-structured summary covering "
            "all major topics found in the passages (introduction, methodology, results, "
            "conclusions, key findings).\n"
            "4. If the answer is not present in the sources, respond ONLY with:\n"
            "   \"The provided sources do not contain information about this topic.\"\n"
            "5. Do NOT use general knowledge or make up information.\n\n"
            "## Source Passages\n" + ctx_text
        )

    if intent == "web_search":
        lines = []
        for i, r in enumerate(web_results, 1):
            title = r.get("title", "No title")
            href = r.get("href", "")
            body = r.get("body", "")[:400]
            tag = "[OpenAlex]" if r.get("source") == "openalex" else "[Web]"
            lines.append(f"{tag} Result {i}: [{title}]({href})\nSnippet: {body}\n")
        results_text = "\n".join(lines) if lines else "(No search results found.)"
        return (
            "You are a research assistant. The user asked a question and real search "
            "results are provided below.\n"
            "Write a well-structured answer in MARKDOWN that:\n"
            "1. Summarises KEY FINDINGS from the snippets — do NOT merely list titles and links.\n"
            "2. Groups information into short paragraphs or bullet points.\n"
            "3. Uses **bold** for important terms.\n"
            "4. Cites sources INLINE as markdown links [Title](url).\n"
            "   ONLY cite URLs that appear verbatim in the results below.\n"
            "5. Ends with a '## Sources' section listing all referenced links.\n"
            "IMPORTANT: Do NOT invent, fabricate, or hallucinate articles, papers, or URLs.\n\n"
            "## Search Results\n" + results_text
        )

    if intent == "analyze":
        ctx_text = (
            "\n\n".join(c.get("content", "") for c in rag_chunks)
            if rag_chunks
            else ""
        )
        return (
            "You are a data analyst and research reviewer. Provide structured analysis with:\n"
            "- **Key Findings** (bullet points)\n"
            "- **Summary** with markdown tables when comparing items\n"
            + (("\n## Content\n" + ctx_text) if ctx_text else "")
        )

    if intent == "latex":
        return (
            "You are an expert LaTeX code generator. Generate clean, compilable LaTeX.\n"
            "- Wrap code in ```latex blocks\n"
            "- Use amsmath, amssymb, graphicx, booktabs\n"
            "- Add comments for complex constructs"
        )

    if intent == "task":
        parts = []
        if metadata.get("project_id"):
            parts.append(f"\nProject ID: {metadata['project_id']}")
        if metadata.get("completed_tasks"):
            parts.append("Completed: " + ", ".join(metadata["completed_tasks"]))
        if metadata.get("current_tasks"):
            parts.append("Current: " + ", ".join(metadata["current_tasks"]))
        task_ctx = "\n".join(parts) if parts else ""
        return (
            "You are a research project planner. Suggest actionable tasks with:\n"
            "- **Title**, **Priority** (High/Medium/Low), **Estimated Time**\n"
            "- **Description** and **Dependencies**\n" + task_ctx
        )

    # Default: "chat"
    return (
        "You are a helpful AI assistant for scientific research project management. "
        "Respond using Markdown: use **bold** for key terms, bullet points (- item) for lists, "
        "## headers for sections, and `code` for technical terms. Be concise and informative."
    )


# ── Streaming endpoint ────────────────────────────────────────────────────────

@router.post("/chat")
async def chat(request: ChatRequest):
    """Unified chat endpoint with real token-by-token streaming."""

    # Build LangChain message history
    messages: list = []
    for msg in request.messages:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))

    metadata = {
        "project_id": request.project_id,
        "document_ids": request.document_ids,
        "intent_hint": request.intent_hint,
        "web_search_sites": request.web_search_sites,
    }

    # Extract the last user message for search / RAG
    last_user_query = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_query = m.content
            break

    async def event_stream():
        try:
            # ── Step 1: Classify ──────────────────────────────────────────────
            state: dict = {"messages": messages, "metadata": metadata, "context": []}
            classified = await classify_intent(state)
            intent: str = classified.get("intent", "chat")

            rag_chunks: list[dict] = []
            web_results: list[dict] = []

            # When doc_ids are pinned the request MUST go through the RAG
            # pipeline. Specialized agents (analyze/task) have no document
            # context, so override them. "latex" is purely generative and is
            # left alone.
            doc_ids = request.document_ids or []
            logger.info(
                "Pipeline start: intent=%r doc_ids=%d query=%r",
                intent, len(doc_ids), last_user_query[:60],
            )

            if doc_ids and intent in SPECIALIZED_AGENTS and intent != "latex":
                logger.info("doc_ids present — overriding %r → rag", intent)
                intent = "chat"  # becomes "rag" after retrieval below

            if intent not in SPECIALIZED_AGENTS:
                # ── Step 2a: Rewrite query (English keywords for web/embedding) ─
                optimized_query = await _rewrite_query(last_user_query)

                # ── Step 2b: RAG ──────────────────────────────────────────────
                if doc_ids:
                    # Summarisation queries ("tóm tắt", "summarize", …) are
                    # meta-questions that don't match document content by
                    # similarity.  Detect on the RAW Vietnamese query (before
                    # rewriting) and use a broad coverage anchor + high k.
                    if _is_summarize(last_user_query):
                        rag_query = (
                            "main topics key findings methodology "
                            "results conclusions introduction"
                        )
                        rag_k = 25
                        logger.info("Summarise request — using coverage query")
                    else:
                        # Targeted question: English-optimised rewrite gives
                        # better cosine-similarity against document chunks.
                        rag_query = optimized_query
                        rag_k = 8

                    try:
                        rag_chunks = await retrieve_documents(
                            rag_query, k=rag_k, document_ids=doc_ids
                        )
                        logger.info(
                            "RAG: rag_query=%r k=%d → %d chunks",
                            rag_query, rag_k, len(rag_chunks),
                        )
                    except Exception as rag_err:
                        logger.warning("RAG retrieval failed: %s", rag_err)

                # ── Step 2c: Final intent ─────────────────────────────────────
                if doc_ids:
                    intent = "rag"
                elif request.web_search_sites:
                    try:
                        ws_agent = get_agent("web_search")
                        web_results, _ = ws_agent.fetch_results_with_query(
                            optimized_query,
                            request.web_search_sites,
                        )
                        logger.info("Web search returned %d results", len(web_results))
                    except Exception as ws_err:
                        logger.warning("Web search failed: %s", ws_err)

                    if web_results:
                        intent = "web_search"
                    else:
                        intent = "chat"
                        yield (
                            "data: Không tìm thấy kết quả từ web search. "
                            "AI sẽ trả lời dựa trên kiến thức nội tại.\\n\\n\n\n"
                        )
                else:
                    intent = "chat"

            # ── Step 3: Send early META so the FE can show the agent badge ────
            agent = get_agent(intent)
            yield f"data: [META]{json.dumps({'agent': agent.name, 'intent': intent})}\n\n"

            # ── Step 4: Build prompt and stream tokens ─────────────────────────
            system_prompt = _build_system_prompt(intent, rag_chunks, web_results, metadata)
            llm = get_llm()
            llm_messages = [SystemMessage(content=system_prompt)] + messages

            async for chunk in llm.astream(llm_messages):
                token = (
                    chunk.content
                    if isinstance(chunk.content, str)
                    else str(chunk.content)
                )
                if token:
                    yield f"data: {token.replace(chr(10), chr(92) + 'n')}\n\n"

            # ── Step 5: Final META with sources (to be persisted by the FE) ───
            sources: list[dict] = []
            if intent == "rag":
                # Deduplicate by source name — keep the first (highest-score)
                # snippet per file so the FE doesn't show the same file 25×.
                seen_sources: dict[str, str] = {}
                for c in rag_chunks:
                    src = c.get("source", "")
                    if src and src not in seen_sources:
                        seen_sources[src] = c.get("content", "")[:600]
                sources = [
                    {"source": src, "snippet": snippet}
                    for src, snippet in seen_sources.items()
                ]
            elif intent == "web_search":
                sources = [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "authors": r.get("authors") or None,
                        "year": r.get("year") or None,
                        "snippet": r.get("body", "")[:200] or None,
                    }
                    for r in web_results
                ]

            yield f"data: [META]{json.dumps({'agent': agent.name, 'intent': intent, 'sources': sources})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as exc:
            logger.error("Chat stream error: %s", exc, exc_info=True)
            yield f"data: Error: {exc!s}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Non-streaming endpoint (kept for backward compat) ─────────────────────────

@router.post("/chat/sync", response_model=ChatResponse)
async def chat_sync(request: ChatRequest):
    """Non-streaming version — returns full response at once via LangGraph."""
    from app.agents.router import get_graph

    messages: list = []
    for msg in request.messages:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))

    state = {
        "messages": messages,
        "metadata": {
            "project_id": request.project_id,
            "document_ids": request.document_ids,
            "intent_hint": request.intent_hint,
        },
        "context": [],
    }

    graph = get_graph()
    result = await graph.ainvoke(state)

    agent_output = result.get("agent_output", {})
    return ChatResponse(
        output=AgentOutput(**agent_output),
        intent=result.get("intent"),
    )
