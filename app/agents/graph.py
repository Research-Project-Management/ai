"""
LangGraph Supervisor Pipeline — The Brain of Flux-AI
=====================================================

Architecture: Supervisor → Worker Agents

The Supervisor is the central orchestrator that:
1. Analyzes the user's request and decides the best route
2. Delegates to specialized worker nodes (RAG, Web, etc.)
3. Gathers context and judges quality before response generation

Worker Nodes:
- **rag_node**: Retrieves documents from Qdrant → optional rerank → RagJudge
- **web_research_node**: DuckDuckGo + OpenAlex + Tavily → passage building
- Specialized agents (latex, analyze, task) bypass the pipeline entirely
  and are streamed directly by chat.py

Extensibility:
- Add a new agent: ``@register_agent`` + import in ``__init__.py``
- The Supervisor auto-discovers it via the registry
- No changes needed in this file

Graph Flow::

    supervisor ──┬── rag_node ──┬── web_research_node ── END
                 │              └── END
                 ├── web_research_node ── END
                 └── END (specialized / chat / greeting)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph

from app.agents.registry import agent_descriptions
from app.agents.schemas import RagJudge, RouteDecision
from app.config import get_llm, get_router_llm, get_settings

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────

# Routes that are internal to the pipeline (not offered to LLM for classification)
_INTERNAL_ROUTES = {"rag", "web_search", "end"}

# Greeting / small-talk patterns (Vietnamese + English)
_GREETING_RE = re.compile(
    r"^\s*(xin\s*chào|chào|hello|hi|hey|cảm\s*ơn|thanks|tks|"
    r"ok|okay|oke|được|ừ|uh|hmm)\s*[!.?]*\s*$",
    re.IGNORECASE,
)

# Summarisation detection — Vietnamese + English keywords
_SUMMARIZE_RE = re.compile(
    r"tóm[\s\-]?tắt|tổng[\s\-]?hợp|tổng[\s\-]?quan|toàn[\s\-]?bộ"
    r"|nội[\s\-]?dung|mô[\s\-]?tả"
    r"|summarize|summary|overview|what.{0,10}(in|about|contain)",
    re.IGNORECASE,
)


# ── Pipeline State ───────────────────────────────────────────────────────────

class PipelineState(TypedDict, total=False):
    """State that flows through the Supervisor Pipeline.

    Each node reads what it needs and returns ONLY the fields it modifies.
    Unmentioned fields are preserved automatically by LangGraph.
    """

    # Input
    messages: list[BaseMessage]

    # Routing
    route: str  # Current routing decision
    intent: str  # Final resolved intent for system prompt

    # Context (produced by worker nodes)
    rag_context: str  # Formatted [D#] blocks from RAG retrieval
    web_context: str  # Formatted [P#] blocks from web research
    sources: list[dict[str, Any]]  # Structured sources for FE META event

    # Query processing
    query_original: str  # Original user query (may be Vietnamese)
    query_en: str  # English translation (for search/rerank)

    # Configuration (set at pipeline start, read-only during execution)
    web_search_enabled: bool
    document_ids: list[str]
    web_search_sites: list[str]
    metadata: dict[str, Any]
    user_id: str  # Authenticated user — used for RAG isolation
    chat_id: str  # Current chat session — used for RAG isolation

    # Quick replies (greeting/end — chat.py sends this directly)
    early_reply: str


# ── Reranker (lazy-loaded, optional) ─────────────────────────────────────────

_reranker = None
_reranker_loaded = False


def _get_reranker():
    """Lazy-load the CrossEncoder reranker. Returns None if unavailable."""
    global _reranker, _reranker_loaded
    if _reranker_loaded:
        return _reranker
    _reranker_loaded = True

    settings = get_settings()
    if not settings.enable_reranking:
        logger.info("Reranking disabled via config.")
        return None

    try:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(settings.cross_encoder_model)
        logger.info("CrossEncoder loaded: %s", settings.cross_encoder_model)
    except ImportError:
        logger.warning(
            "sentence-transformers not installed — reranking disabled. "
            "Install with: pip install sentence-transformers"
        )
    except Exception as exc:
        logger.warning("Failed to load CrossEncoder: %s", exc)

    return _reranker


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_last_user_query(messages: list[BaseMessage]) -> str:
    """Extract the last human message content."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _translate_to_english(text: str) -> str:
    """Translate query to English for better search/rerank quality."""
    try:
        llm = get_llm(temperature=0)
        response = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Translate the following text to natural English. "
                        "If already English, return unchanged. Return ONLY the English text."
                    ),
                },
                {"role": "user", "content": text},
            ]
        )
        return (response.content or "").strip() or text
    except Exception as exc:
        logger.warning("Translation failed: %s", exc)
        return text


def _is_summarize(text: str) -> bool:
    """Detect summarisation requests in Vietnamese + English."""
    return bool(_SUMMARIZE_RE.search(text))


def _rerank_chunks(query: str, chunks: list[dict], top_k: int = 10) -> list[dict]:
    """Rerank retrieved chunks using CrossEncoder (if available)."""
    reranker = _get_reranker()
    if not reranker or not chunks:
        return chunks[:top_k]

    pairs = [[query, (c.get("content", ""))[:2000]] for c in chunks]
    try:
        scores = reranker.predict(pairs)
        scored = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]
    except Exception as exc:
        logger.warning("Rerank failed: %s — using original order", exc)
        return chunks[:top_k]


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
        metadata = c.get("metadata", {})
        page = metadata.get("page")
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


def _format_web_context(results: list[dict], max_results: int = 12) -> str:
    """Format web search results into [P#] blocks for the LLM context."""
    lines = ["WEB RESEARCH RESULTS:"]
    for i, r in enumerate(results[:max_results], 1):
        title = r.get("title", "No title")
        href = r.get("href", r.get("url", ""))
        body = (r.get("body", r.get("content", "")))[:800]
        source_tag = "[OpenAlex]" if r.get("source") == "openalex" else "[Web]"

        lines.append(f"[P{i}] {source_tag}")
        lines.append(f"Title: {title}")
        if href:
            lines.append(f"URL: {href}")
        lines.append(f"TEXT: {body}")
        lines.append("")

    return "\n".join(lines).strip()


def _build_agent_catalog() -> str:
    """Auto-generate agent descriptions for the Supervisor prompt.

    This is what makes the Supervisor extensible: new agents registered
    via ``@register_agent`` are automatically included here.
    """
    catalog_lines = []
    for agent in agent_descriptions():
        if agent["name"] not in _INTERNAL_ROUTES:
            catalog_lines.append(f"- **{agent['name']}**: {agent['description']}")
    return "\n".join(catalog_lines)


# ── Supervisor Prompt ────────────────────────────────────────────────────────

SUPERVISOR_SYSTEM_PROMPT = """\
You are the **Supervisor** — the central orchestrator of a multi-agent AI system \
called Flux-AI. Your job is to analyze the user's message and route it to the \
most appropriate handler.

## Available Specialized Agents
{agent_catalog}

## Pipeline Routes (handled automatically)
- **rag**: Query the user's uploaded documents via vector search. Use when the \
user asks about content in their documents.
- **web_search**: Search the internet (DuckDuckGo + OpenAlex academic database). \
Use when the user needs current information, research papers, or anything beyond \
their uploaded documents.

## Routing Rules  
1. If the user asks about their **workspace content** (projects, members, recent items, activity feed) → **action**
2. If the user wants to **create, update, or delete** something (task, goal, cycle, sticky, page) → **action**
3. If the user asks about **project status, progress, tasks, team** → **action**
4. If the user asks to **search** within their workspace → **action**
5. If the user references their **uploaded documents or files** → **rag**
6. If the user needs **external/recent information** or asks to search the web → **web_search**
7. If the request clearly matches another specialized agent (latex, analyze, task planner) → that agent's name
8. If it's a **general knowledge question** with no workspace context needed → **chat**

## Action Agent Triggers (always route to **action** for these)
- "workspace có gì", "workspace của tôi", "có những project nào"
- "tạo task", "tạo goal", "tạo cycle", "tạo sticky"
- "member trong workspace", "ai trong team"
- "tiến độ project", "project status", "có gì mới"
- "create task", "create goal", "show projects", "workspace overview"
- Any CRUD operation on workspace entities

## Runtime Status
- Web search: {web_status}
- User documents pinned: {docs_status}

IMPORTANT: Respond with ONLY a JSON object: {{"route": "<agent_name>", "reasoning": "brief reason"}}
"""


# ═══════════════════════════════════════════════════════════════════════════
# NODE 1: SUPERVISOR
# ═══════════════════════════════════════════════════════════════════════════

async def supervisor_node(state: PipelineState) -> dict:
    """Route the request to the right worker node.

    Decision flow:
    1. Fast paths — empty query, greeting/thanks → end
    2. Explicit ``intent_hint`` from the FE → respect it
    3. ``document_ids`` present → force ``rag``
    4. LLM classification (structured output) → best agent
    5. Hard constraints (web disabled → reroute, etc.)
    """
    logger.info("════ Supervisor ════")

    messages = state.get("messages", [])
    metadata = state.get("metadata", {})
    query_original = _get_last_user_query(messages)

    web_search_enabled = state.get("web_search_enabled", False)
    document_ids = state.get("document_ids", [])
    kb_available = bool(document_ids)

    # ─── Fast path: empty query ───
    if not query_original.strip():
        logger.info("Empty query → end")
        return {
            "route": "end",
            "intent": "chat",
            "query_original": "",
            "early_reply": "Chào bạn! Bạn muốn hỏi gì?",
        }

    # ─── Fast path: greeting / small-talk ───
    if _GREETING_RE.match(query_original.strip()):
        logger.info("Greeting detected → end")
        return {
            "route": "end",
            "intent": "chat",
            "query_original": query_original,
            "early_reply": "Xin chào! Bạn muốn hỏi gì về bài báo/tài liệu?",
        }

    # ─── Explicit intent hint from FE ───
    intent_hint = metadata.get("intent_hint")
    if intent_hint:
        known_agents = {a["name"] for a in agent_descriptions()}
        all_routes = known_agents | {"rag", "web_search"}
        if intent_hint in all_routes:
            logger.info("Intent hint → %s", intent_hint)
            return {
                "route": intent_hint,
                "intent": intent_hint,
                "query_original": query_original,
            }

    # ─── Force RAG when document_ids are pinned ───
    if document_ids:
        logger.info("doc_ids pinned → rag")
        return {
            "route": "rag",
            "intent": "rag",
            "query_original": query_original,
        }

    # ─── LLM Classification ───
    agent_catalog = _build_agent_catalog()
    system_prompt = SUPERVISOR_SYSTEM_PROMPT.format(
        agent_catalog=agent_catalog,
        web_status="ENABLED" if web_search_enabled else "DISABLED",
        docs_status=f"YES ({len(document_ids)} files)" if kb_available else "NONE",
    )

    try:
        llm = get_router_llm()
        response = await llm.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query_original},
            ]
        )

        content = response.content if isinstance(response.content, str) else str(response.content)

        # Parse JSON response (handle markdown code fences)
        clean = content.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(clean)
        route = parsed.get("route", "chat")
        reasoning = parsed.get("reasoning", "")
        logger.info("LLM classification: route=%s reasoning=%s", route, reasoning)

    except Exception as exc:
        logger.warning("Supervisor LLM failed: %s → fallback", exc)
        if kb_available:
            route = "rag"
        elif web_search_enabled:
            route = "web_search"
        else:
            route = "chat"

    # ─── Validate route ───
    known_agents = {a["name"] for a in agent_descriptions()}
    valid_routes = known_agents | {"rag", "web_search"}
    if route not in valid_routes:
        route = "chat"

    # ─── Hard constraints ───
    if not web_search_enabled and route == "web_search":
        route = "rag" if kb_available else "chat"

    if not kb_available and route == "rag":
        route = "web_search" if web_search_enabled else "chat"

    # ─── Web search override ───
    # If web is enabled and LLM chose "chat" (general knowledge fallback),
    # we can do better by actually searching the web. This handles cases where
    # the user explicitly turned on web search but asked a general question.
    if web_search_enabled and route == "chat":
        logger.info("Web enabled + chat route → upgrading to web_search")
        route = "web_search"

    logger.info("Supervisor final: route=%s", route)
    return {
        "route": route,
        "intent": route,
        "query_original": query_original,
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE 2: RAG RETRIEVAL + JUDGE
# ═══════════════════════════════════════════════════════════════════════════

async def rag_node(state: PipelineState) -> dict:
    """Retrieve documents, optionally rerank, and judge sufficiency.

    Flow:
    1. Retrieve chunks from Qdrant (filtered by doc_ids)
    2. Rerank with CrossEncoder (if available)
    3. Format as [D#] context blocks
    4. RagJudge: is this enough to answer? → 'done' or 'web'
    """
    logger.info("════ RAG Node ════")

    query_original = state.get("query_original", "")
    document_ids = state.get("document_ids", [])
    web_search_enabled = state.get("web_search_enabled", False)

    if not query_original:
        return {"rag_context": "", "route": "done", "sources": []}

    # Translate for reranking
    query_en = await asyncio.to_thread(_translate_to_english, query_original)

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
    try:
        from app.rag.retriever import retrieve_documents

        # Read isolation context from pipeline state
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
        chunks = []

    if not chunks:
        logger.info("No chunks → route to web" if web_search_enabled else "No chunks → done")
        return {
            "rag_context": "",
            "query_en": query_en,
            "route": "web" if web_search_enabled else "done",
            "sources": [],
        }

    # Rerank (optional)
    chunks = await asyncio.to_thread(
        _rerank_chunks, query_en, chunks, min(10, len(chunks))
    )
    logger.info("After rerank: %d chunks", len(chunks))

    # Format context
    rag_context = _format_rag_context(chunks)

    # Build sources for FE
    seen_sources: dict[str, str] = {}
    for c in chunks:
        src = c.get("source", "")
        if src and src not in seen_sources:
            seen_sources[src] = c.get("content", "")[:400]
    sources = [
        {"source": src, "snippet": snippet}
        for src, snippet in seen_sources.items()
    ]

    # ─── RagJudge: is context sufficient? ───
    next_route = "done"
    try:
        judge_llm = get_router_llm()
        judge_prompt = (
            "You are a judge evaluating whether retrieved text is sufficient "
            "and relevant to answer the question.\n\n"
            "Criteria:\n"
            "- Directly addresses the main question\n"
            "- Enough detail for a clear answer\n"
            "- Specific and relevant, not vague\n\n"
            "Respond with ONLY JSON: {\"sufficient\": true/false, \"reasoning\": \"brief\"}"
        )
        judge_response = await judge_llm.ainvoke(
            [
                {"role": "system", "content": judge_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Question: {query_original}\n\n"
                        f"Retrieved context:\n{rag_context[:8000]}\n\n"
                        "Is this sufficient?"
                    ),
                },
            ]
        )

        judge_content = judge_response.content if isinstance(judge_response.content, str) else str(judge_response.content)
        judge_clean = judge_content.strip()
        if judge_clean.startswith("```"):
            judge_clean = judge_clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        verdict = json.loads(judge_clean)
        sufficient = verdict.get("sufficient", True)
        logger.info("RagJudge: sufficient=%s reasoning=%s", sufficient, verdict.get("reasoning", ""))

        if not sufficient and web_search_enabled:
            next_route = "web"

    except Exception as exc:
        logger.warning("RagJudge failed: %s → treat as sufficient", exc)

    return {
        "rag_context": rag_context,
        "query_en": query_en,
        "route": next_route,
        "sources": sources,
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE 3: WEB RESEARCH
# ═══════════════════════════════════════════════════════════════════════════

async def web_research_node(state: PipelineState) -> dict:
    """Search the web for relevant information.

    Uses:
    - DuckDuckGo (general web, filtered by academic sites)
    - OpenAlex (academic database)
    - Tavily (if API key configured — higher quality snippets)

    All search functions are sync; we run them in a thread pool.
    """
    logger.info("════ Web Research Node ════")

    query_original = state.get("query_original", "")
    web_search_sites = state.get("web_search_sites", [])

    if not query_original:
        return {"web_context": "", "route": "done"}

    # Translate to English for search
    query_en = state.get("query_en") or await asyncio.to_thread(
        _translate_to_english, query_original
    )

    # ─── Search using existing web_search_agent functions ───
    try:
        from app.agents.web_search_agent import _search

        results = await asyncio.to_thread(
            _search, query_en, web_search_sites or None
        )
        logger.info("DDG + OpenAlex: %d results", len(results))
    except Exception as exc:
        logger.warning("DDG/OpenAlex search failed: %s", exc)
        results = []

    # ─── Tavily search (if configured) ───
    settings = get_settings()
    if settings.tavily_api_key:
        try:
            tavily_results = await asyncio.to_thread(
                _search_tavily, query_en, settings.tavily_api_key, web_search_sites
            )
            logger.info("Tavily: %d results", len(tavily_results))

            # Merge + dedup by URL
            seen_urls = {r.get("href", r.get("url", "")) for r in results if r.get("href") or r.get("url")}
            for tr in tavily_results:
                url = tr.get("href", tr.get("url", ""))
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    results.append(tr)
        except Exception as exc:
            logger.warning("Tavily search failed: %s", exc)

    if not results:
        logger.info("No web results found")
        return {"web_context": "", "query_en": query_en, "route": "done"}

    # Format context
    web_context = _format_web_context(results)

    # Build sources for FE
    existing_sources = state.get("sources", [])
    web_sources = [
        {
            "title": r.get("title", ""),
            "url": r.get("href", r.get("url", "")),
            "authors": r.get("authors") or None,
            "year": r.get("year") or None,
            "snippet": (r.get("body", r.get("content", "")))[:200] or None,
        }
        for r in results
    ]

    return {
        "web_context": web_context,
        "query_en": query_en,
        "route": "done",
        "sources": existing_sources + web_sources,
    }


def _search_tavily(
    query: str,
    api_key: str,
    include_domains: list[str] | None = None,
    max_results: int = 5,
) -> list[dict]:
    """Search using Tavily API (higher quality snippets than DDG).

    Uses the REST API directly via httpx — no extra dependency needed.
    """
    try:
        import httpx

        payload: dict[str, Any] = {
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        if include_domains:
            payload["include_domains"] = include_domains

        resp = httpx.post(
            "https://api.tavily.com/search",
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "href": item.get("url", ""),
                    "body": (item.get("content", ""))[:800],
                    "source": "tavily",
                }
            )
        return results
    except Exception as exc:
        logger.warning("Tavily API call failed: %s", exc)
        return []


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def _route_after_supervisor(state: PipelineState) -> str:
    """Conditional edge after supervisor: which node to visit next."""
    route = state.get("route", "chat")
    if route == "rag":
        return "rag_node"
    if route == "web_search":
        return "web_research_node"
    # Everything else (specialized agents, chat, end) → exit the graph.
    # chat.py handles these by streaming with the appropriate system prompt.
    return END


def _route_after_rag(state: PipelineState) -> str:
    """Conditional edge after RAG: go to web or finish."""
    route = state.get("route", "done")
    if route == "web":
        return "web_research_node"
    return END


def build_pipeline() -> StateGraph:
    """Build and compile the Supervisor Pipeline graph."""
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("web_research_node", web_research_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Conditional edges
    graph.add_conditional_edges("supervisor", _route_after_supervisor)
    graph.add_conditional_edges("rag_node", _route_after_rag)
    graph.add_edge("web_research_node", END)

    return graph.compile()


# ── Singleton ────────────────────────────────────────────────────────────────

_pipeline = None


def get_pipeline():
    """Get or create the compiled pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
        logger.info("Supervisor Pipeline compiled")
    return _pipeline
