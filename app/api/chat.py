"""
Chat endpoint — unified gateway to the multi-agent system.

Supports real token-by-token streaming and non-streaming responses.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.agents.router import classify_intent
from app.agents.registry import get_agent
from app.config import get_llm
from app.models import AgentOutput, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Unified chat endpoint with REAL token-by-token streaming.

    1. Classify intent (fast, non-streaming)
    2. Stream the LLM response token-by-token via SSE
    """
    # Build LangChain messages from request
    messages = []
    for msg in request.messages:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))

    # Build initial state for intent classification
    state = {
        "messages": messages,
        "metadata": {
            "project_id": request.project_id,
            "document_ids": request.document_ids,
            "intent_hint": request.intent_hint,
            "web_search_sites": request.web_search_sites,
        },
        "context": [],
    }

    # Pre-fetch RAG context if needed
    if request.document_ids:
        try:
            from app.rag.retriever import retrieve_documents

            last_msg = request.messages[-1].content if request.messages else ""
            results = await retrieve_documents(last_msg, document_ids=request.document_ids)
            state["context"] = results
        except Exception as e:
            logger.warning("Failed to retrieve documents: %s", e)

    async def event_stream():
        """SSE stream with real token-by-token LLM output."""
        try:
            # Step 1: Classify intent (fast)
            classified = await classify_intent(state)
            intent = classified.get("intent", "chat")
            agent = get_agent(intent)

            logger.info("Streaming with agent: %s", agent.name)

            # Step 2: Send META immediately so FE can show agent label
            meta_early = {"agent": agent.name, "intent": intent}
            yield f"data: [META]{json.dumps(meta_early)}\n\n"

            # Step 3: Pre-execute side-effects (web search, RAG) before streaming
            if intent == "web_search":
                try:
                    ws_results, ws_query = agent.fetch_results(state)
                    state["web_search_results"] = ws_results
                    state["web_search_query"] = ws_query
                    logger.info("Web search returned %d results", len(ws_results))
                except Exception as ws_err:
                    logger.warning("Web search pre-step failed: %s", ws_err)
                    state["web_search_results"] = []

                # If search returned nothing, tell the user immediately — don't let the LLM hallucinate
                if not state.get("web_search_results"):
                    yield "data: Không tìm thấy kết quả nào từ web search. Vui lòng thử lại với từ khóa khác hoặc tắt web search để nhận câu trả lời từ kiến thức nội tại của AI.\n\n"
                    yield f"data: [META]{json.dumps({'agent': agent.name, 'intent': intent, 'sources': []})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

            # Build the LLM prompt based on agent type
            llm = get_llm()
            llm_messages = _build_agent_messages(agent, state)

            # Step 4: Stream tokens from LLM
            full_content = ""
            async for chunk in llm.astream(llm_messages):
                token = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                if token:
                    full_content += token
                    # Escape newlines so they survive SSE line-based transport
                    # (a bare \n in a `data:` field breaks the SSE framing)
                    safe_token = token.replace("\n", "\\n")
                    yield f"data: {safe_token}\n\n"

            # Step 4: Send final metadata with sources
            meta = {
                "agent": agent.name,
                "intent": intent,
                "sources": [],
            }

            # Add RAG sources if available
            if intent == "rag" and state.get("context"):
                meta["sources"] = [
                    {"source": c.get("source", ""), "score": c.get("score", 0)}
                    for c in state["context"]
                ]

            # Add web search sources if available
            if intent == "web_search" and state.get("web_search_results"):
                meta["sources"] = [
                    {"title": r.get("title", ""), "url": r.get("href", "")}
                    for r in state["web_search_results"]
                ]

            yield f"data: [META]{json.dumps(meta)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error("Chat stream error: %s", e, exc_info=True)
            yield f"data: Error: {str(e)}\n\n"
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


def _build_agent_messages(agent, state: dict) -> list:
    """Build LLM messages based on the agent type and state."""
    from langchain_core.messages import SystemMessage

    messages = state.get("messages", [])
    context_chunks = state.get("context", [])

    # Agent-specific system prompts
    prompts = {
        "chat": (
            "You are a helpful AI assistant for scientific research project management. "
            "Respond using Markdown: use **bold** for key terms, bullet points (- item) for lists, "
            "## headers for sections, and `code` for technical terms. Be concise and informative."
        ),
        "rag": (
            "You are a research assistant. Answer using ONLY the provided context. "
            "Cite sources. If context doesn't contain the answer, say so.\n\n"
            "## Retrieved Context\n"
            + (
                "\n\n---\n\n".join(
                    f"[Source: {c.get('source', '?')}]\n{c.get('content', '')}"
                    for c in context_chunks
                )
                if context_chunks
                else "(No documents found.)"
            )
        ),
        "analyze": (
            "You are a data analyst and research reviewer. Provide structured analysis with:\n"
            "- **Key Findings** (bullet points)\n"
            "- **Summary** with markdown tables if comparing items\n"
            + (
                "\n## Content\n"
                + "\n\n".join(c.get("content", "") for c in context_chunks)
                if context_chunks
                else ""
            )
        ),
        "latex": (
            "You are an expert LaTeX code generator. Generate clean, compilable LaTeX.\n"
            "- Wrap code in ```latex blocks\n"
            "- Use amsmath, amssymb, graphicx, booktabs\n"
            "- Add comments for complex constructs"
        ),
        "task": (
            "You are a research project planner. Suggest actionable tasks with:\n"
            "- **Title**, **Priority** (High/Medium/Low), **Estimated Time**\n"
            "- **Description** and **Dependencies**\n"
            + _build_task_context(state.get("metadata", {}))
        ),
        "web_search": (
            "You are a research assistant. The user asked a question and real search results are provided below.\n"
            "Write a well-structured answer in MARKDOWN that:\n"
            "1. Summarises the KEY FINDINGS and main points from the snippets — do NOT just list titles and links.\n"
            "2. Groups related information into short paragraphs or bullet points using markdown (- item).\n"
            "3. Uses **bold** for important terms.\n"
            "4. Cites sources INLINE using markdown links like [Paper Title](url) — NOT [1], [2] footnotes, NOT [Link] placeholders.\n"
            "   ONLY cite URLs that appear verbatim in the search results below. Never invent or guess URLs.\n"
            "5. Ends with a '## Sources' section listing all referenced links as markdown links: [Title](url).\n"
            "IMPORTANT: Do NOT invent, fabricate, or hallucinate any articles, papers, or URLs not present in the search results.\n"
            "If the provided results are not relevant or sufficient, say exactly that — do not make up content.\n\n"
            "## Search Results\n"
            + _format_web_results(state.get("web_search_results", []))
        ),
    }

    system_prompt = prompts.get(agent.name, prompts["chat"])
    return [SystemMessage(content=system_prompt)] + list(messages)


def _format_web_results(results: list) -> str:
    if not results:
        return "(No search results found.)"
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        href = r.get("href", "")
        body = r.get("body", "")[:400]
        source_tag = "[OpenAlex]" if r.get("source") == "openalex" else "[Web]"
        # Provide title + url together so the LLM can easily produce [Title](url) citations
        lines.append(f"{source_tag} Result {i}: [{title}]({href})\nSnippet: {body}\n")
    return "\n".join(lines)


def _build_task_context(metadata: dict) -> str:
    """Build project context for task agent."""
    parts = []
    if metadata.get("project_id"):
        parts.append(f"\nProject ID: {metadata['project_id']}")
    if metadata.get("completed_tasks"):
        parts.append("Completed: " + ", ".join(metadata["completed_tasks"]))
    if metadata.get("current_tasks"):
        parts.append("Current: " + ", ".join(metadata["current_tasks"]))
    return "\n".join(parts) if parts else ""


@router.post("/chat/sync", response_model=ChatResponse)
async def chat_sync(request: ChatRequest):
    """Non-streaming version — returns full response at once."""
    from app.agents.router import get_graph

    messages = []
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
