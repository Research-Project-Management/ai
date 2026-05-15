"""
Supervisor Node — Routes user requests to the correct pipeline worker.

Decision flow:
  1. Fast paths — empty query, greeting/thanks → end with early_reply
  2. Explicit intent_hint from the FE → respect it
  3. document_ids present → force rag
  4. LLM classification (JSON structured output) → best route
  5. Hard constraints (web disabled → reroute, etc.)
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import BaseMessage, HumanMessage

from app.agents.registry import agent_descriptions
from app.agents.pipeline.state import PipelineState
from app.config import get_router_llm

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────

_INTERNAL_ROUTES = {"rag", "web_search", "end"}

_GREETING_RE = re.compile(
    r"^\s*(xin\s*chào|chào|hello|hi|hey|cảm\s*ơn|thanks|tks|"
    r"ok|okay|oke|được|ừ|uh|hmm)\s*[!.?]*\s*$",
    re.IGNORECASE,
)

# ── Agent catalog cache ────────────────────────────────────────────────────

_AGENT_CATALOG_CACHE: str | None = None


def _build_agent_catalog() -> str:
    """Auto-generate agent descriptions for the Supervisor prompt.

    New agents registered via @register_agent are automatically included.
    Call invalidate_agent_catalog_cache() after dynamic agent registration.
    """
    global _AGENT_CATALOG_CACHE
    if _AGENT_CATALOG_CACHE is None:
        catalog_lines = [
            f"- **{a['name']}**: {a['description']}"
            for a in agent_descriptions()
            if a["name"] not in _INTERNAL_ROUTES
        ]
        _AGENT_CATALOG_CACHE = "\n".join(catalog_lines)
    return _AGENT_CATALOG_CACHE


def invalidate_agent_catalog_cache() -> None:
    """Force catalog rebuild on next supervisor call.

    Call this after dynamically registering a new agent at runtime.
    """
    global _AGENT_CATALOG_CACHE
    _AGENT_CATALOG_CACHE = None
    logger.info("Agent catalog cache invalidated — will rebuild on next request.")


# ── Supervisor Prompt ──────────────────────────────────────────────────────

_SUPERVISOR_PROMPT = """\
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


# ── Helpers ────────────────────────────────────────────────────────────────

def _get_last_user_query(messages: list[BaseMessage]) -> str:
    """Extract the last human message content."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


# ── Node ───────────────────────────────────────────────────────────────────

async def supervisor_node(state: PipelineState) -> dict:
    """Route the request to the right worker node."""
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
        greeting = (
            "Chào bạn! Bạn muốn hỏi gì về tài liệu đã upload?"
            if kb_available
            else "Chào bạn! Tôi có thể giúp bạn quản lý workspace, tìm kiếm thông tin, hoặc trả lời câu hỏi."
        )
        return {"route": "end", "intent": "chat", "query_original": "", "early_reply": greeting}

    # ─── Fast path: greeting / small-talk ───
    if _GREETING_RE.match(query_original.strip()):
        logger.info("Greeting detected → end")
        if kb_available:
            reply = "Xin chào! Bạn muốn hỏi gì về tài liệu đã upload?"
        elif metadata.get("workspace_id"):
            reply = "Xin chào! Tôi có thể giúp bạn quản lý task, project, cycle, hay tìm kiếm trong workspace."
        else:
            reply = "Xin chào! Tôi có thể giúp gì cho bạn?"
        return {"route": "end", "intent": "chat", "query_original": query_original, "early_reply": reply}

    # ─── Explicit intent hint from FE ───
    intent_hint = metadata.get("intent_hint")
    if intent_hint:
        known_agents = {a["name"] for a in agent_descriptions()}
        all_routes = known_agents | {"rag", "web_search"}
        if intent_hint in all_routes:
            logger.info("Intent hint → %s", intent_hint)
            return {"route": intent_hint, "intent": intent_hint, "query_original": query_original}

    # ─── Avoid forcing RAG ───
    # Previously we forced RAG here if document_ids was set, but that prevented
    # users from performing actions (like creating tasks) while files were uploaded.
    # We now let the LLM classifier decide.

    # ─── LLM Classification ───
    agent_catalog = _build_agent_catalog()
    system_prompt = _SUPERVISOR_PROMPT.format(
        agent_catalog=agent_catalog,
        web_status="ENABLED" if web_search_enabled else "DISABLED",
        docs_status=f"YES ({len(document_ids)} files)" if kb_available else "NONE",
    )

    route = "chat"
    try:
        llm = get_router_llm()
        response = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_original},
        ])

        content = response.content if isinstance(response.content, str) else str(response.content)
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

    # ─── Web search upgrade ───
    # If web is enabled and LLM chose "chat", upgrade to web_search for
    # richer answers (user explicitly enabled web search mode).
    if web_search_enabled and route == "chat":
        logger.info("Web enabled + chat route → upgrading to web_search")
        route = "web_search"

    logger.info("Supervisor final: route=%s", route)
    return {"route": route, "intent": route, "query_original": query_original}
