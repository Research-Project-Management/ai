"""
Chat endpoint — powered by the Supervisor Pipeline.

Architecture:
  1. Build initial PipelineState from the incoming request
  2. Run the LangGraph Supervisor Pipeline (supervisor → rag → web)
  3. Stream the answer tokens via LLM with the gathered context
  4. Emit SSE events compatible with the existing frontend

SSE Protocol:
  - ``data: [META]{...}\n\n``    — agent badge + progress
  - ``data: <token>\n\n``        — answer tokens
  - ``data: [ACTION]{...}\n\n``  — tool execution events (action agent only)
  - ``data: [META]{...}\n\n``    — final metadata with sources
  - ``data: [DONE]\n\n``         — stream end
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.graph import PipelineState, get_pipeline
from app.agents.orchestrator import stream_orchestrator
from app.agents.registry import get_agent
from app.config import get_agent_llm, get_llm
from app.models import AgentOutput, ChatRequest, ChatResponse
from app.tools._client import set_trace_id, get_trace_id, set_user_message

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


from app.agents.utils import _extract_token


# ── System Prompt Builder ────────────────────────────────────────────────────

def _build_system_prompt(
    intent: str,
    rag_context: str,
    web_context: str,
    metadata: dict,
) -> str:
    """Build the system prompt for the final LLM answer generation.

    This function is called AFTER the pipeline has gathered context.
    The prompt guides the LLM to produce a high-quality, cited answer.
    """

    if intent == "rag":
        ctx_text = rag_context or "(No content could be retrieved from the selected sources.)"
        return (
            "You are a research assistant specializing in answering questions STRICTLY "
            "based on the user's uploaded source documents provided below.\n\n"
            "## RULES\n"
            "1. **Faithfulness**: ONLY use information that appears in the Source Passages.\n"
            "2. **Citations**: Cite facts inline as **[Source: filename]**.\n"
            "3. **Summarization**: For summary requests, produce a well-structured summary "
            "covering all major topics (introduction, methodology, results, conclusions, "
            "key findings).\n"
            "4. **Honesty**: If the answer is NOT in the sources, respond ONLY with:\n"
            '   "The provided sources do not contain information about this topic."\n'
            "5. **No Hallucination**: Do NOT use general knowledge or invent information.\n"
            "6. **Language**: Respond in the same language as the user's question.\n"
            "7. **Format**: Use Markdown — **bold** for key terms, bullet points for lists, "
            "## headers for sections.\n\n"
            "## Source Passages\n" + ctx_text
        )

    if intent == "web_search":
        ctx_text = web_context or "(No search results found.)"
        return (
            "You are a research assistant. The user asked a question and real search "
            "results are provided below.\n\n"
            "## RULES\n"
            "1. **Synthesize**: Summarise KEY FINDINGS from the snippets — do NOT merely "
            "list titles and links.\n"
            "2. **Structure**: Group information into short paragraphs or bullet points.\n"
            "3. **Bold Key Terms**: Use **bold** for important concepts.\n"
            "4. **Cite Sources**: Use inline markdown links [Title](url).\n"
            "   ONLY cite URLs that appear verbatim in the results.\n"
            "5. **Sources Section**: End with a '## Sources' section listing all references.\n"
            "6. **No Hallucination**: Do NOT invent articles, papers, or URLs.\n"
            "7. **Language**: Respond in the same language as the user's question.\n\n"
            "## Search Results\n" + ctx_text
        )

    if intent == "analyze":
        return (
            "You are a data analyst and research reviewer. Provide structured analysis with:\n"
            "- **Key Findings** (bullet points)\n"
            "- **Methodology Assessment** (if applicable)\n"
            "- **Strengths & Weaknesses**\n"
            "- **Summary Table** (use markdown table when comparing items)\n"
            "Respond in the same language as the user's question."
            + (("\n\n## Content\n" + rag_context) if rag_context else "")
        )

    if intent == "latex":
        return (
            "You are an expert LaTeX code generator. Generate clean, compilable LaTeX.\n"
            "- Wrap code in ```latex blocks\n"
            "- Use amsmath, amssymb, graphicx, booktabs\n"
            "- Add comments for complex constructs\n"
            "- Ensure the output is valid and compilable"
        )

    if intent == "latex_editor":
        # LaTeX editor agent builds its own detailed system prompt internally.
        # This path is used only for the final streaming pass (non-agent path).
        # In practice, latex_editor is routed to run_agent() like other specialized agents.
        return (
            "You are an expert LaTeX writing assistant embedded in the Flux editor. "
            "Help the user with their LaTeX document. "
            "Wrap any LaTeX code in ```latex blocks. "
            "Respond in the same language as the user."
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
            "- **Description** and **Dependencies**\n"
            "Respond in the same language as the user's question.\n" + task_ctx
        )

    # Default: "chat"
    return (
        "You are a helpful AI assistant for scientific research project management. "
        "Respond using Markdown: use **bold** for key terms, bullet points (- item) for lists, "
        "## headers for sections, and `code` for technical terms. Be concise and informative. "
        "Respond in the same language as the user's question."
    )


# ── Streaming Endpoint ───────────────────────────────────────────────────────

@router.post("/chat")
async def chat(request: ChatRequest):
    """Unified chat endpoint with Supervisor Pipeline + real token streaming."""

    # Generate short trace ID for this request (8 hex chars)
    trace_id = uuid.uuid4().hex[:8]
    set_trace_id(trace_id)
    logger.info("[%s] CHAT user=%s intent=%s", trace_id, request.user_id, request.intent_hint)

    # Extract the last user message for server-side confirmation guards
    last_user_msg = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            last_user_msg = msg.content
            break
    set_user_message(last_user_msg)

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
        "workspace_id": request.workspace_id,
        "user_id": request.user_id,
        "chat_id": request.chat_id,
        # LaTeX editor context — forwarded to latex_editor_agent
        "file_content": request.file_content or "",
        "filename": request.filename or "document.tex",
        "selection": request.selection or "",
        "cursor_context": request.cursor_context or "",
        # Cursor & selection range (v2)
        "cursor_line": request.cursor_line or 1,
        "cursor_column": request.cursor_column or 1,
        "selection_start_line": request.selection_start_line,
        "selection_start_column": request.selection_start_column,
        "selection_end_line": request.selection_end_line,
        "selection_end_column": request.selection_end_column,
        "document_structure": request.document_structure or "",
        "command_hint": request.command_hint or "",
    }

    # Build initial pipeline state
    initial_state: PipelineState = {
        "messages": messages,
        "document_ids": request.document_ids or [],
        "web_search_enabled": bool(request.web_search_sites),
        "web_search_sites": request.web_search_sites or [],
        "metadata": metadata,
        "user_id": request.user_id or "",
        "chat_id": request.chat_id or "",
    }

    async def event_stream():
        try:
            pipeline = get_pipeline()

            # ── Run the Supervisor Pipeline (node by node) ────────────
            final_state: dict = {}

            async for update in pipeline.astream(initial_state, stream_mode="updates"):
                # Each update is {node_name: node_output}
                node_name = list(update.keys())[0]
                node_output = update[node_name]

                # Merge node output into final state
                for key, value in node_output.items():
                    final_state[key] = value

                logger.info("Pipeline node completed: %s", node_name)

                # ── Handle supervisor decisions ───
                if node_name == "supervisor":
                    route = final_state.get("route", "chat")
                    intent = final_state.get("intent", "chat")

                    # Fast exit: greeting / end
                    if route == "end":
                        early_reply = final_state.get("early_reply", "")
                        agent_name = "chat"
                        yield f"data: [META]{json.dumps({'agent': agent_name, 'intent': 'chat'})}\n\n"
                        if early_reply:
                            yield f"data: {early_reply.replace(chr(10), chr(92) + 'n')}\n\n"
                        yield f"data: [META]{json.dumps({'agent': agent_name, 'intent': 'chat', 'sources': []})}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    # ── ACTION AGENT ROUTE (A2A Orchestrator) ───────────────
                    if route == "action":
                        yield f"data: [META]{json.dumps({'agent': 'orchestrator', 'intent': 'action', 'trace_id': get_trace_id()})}\n\n"

                        # Emit early "thinking" so user sees activity immediately —
                        # before orchestrator LLM decides which sub-agent to call.
                        yield f"data: [ACTION]{json.dumps({'type': 'thinking', 'agent': 'orchestrator'})}\n\n"

                        # Stream qua A2A Orchestrator (delegate sang sub-agents)
                        async for event_data in stream_orchestrator(messages, request):
                            yield event_data

                        # Final META
                        yield f"data: [META]{json.dumps({'agent': 'orchestrator', 'intent': 'action', 'sources': []})}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    # ── LATEX EDITOR AGENT ROUTE ─────────────────────────────
                    # Uses real LLM astream (token-by-token) instead of blocking
                    # agent.run() + word-split chunking.
                    if route == "latex_editor":
                        from app.agents.latex_editor_agent import (
                            LATEX_EDITOR_PROMPT,
                            _add_line_numbers,
                            _build_selection_context,
                            _build_selection_info,
                            _truncate_file_content,
                        )
                        yield f"data: [META]{json.dumps({'agent': 'latex_editor', 'intent': 'latex_editor'})}\n\n"

                        # Extract editor context from metadata
                        file_content = metadata.get("file_content", "")
                        filename = metadata.get("filename", "document.tex")
                        selection = metadata.get("selection", "")
                        cursor_context = metadata.get("cursor_context", "")
                        cursor_line = metadata.get("cursor_line") or 1
                        cursor_column = metadata.get("cursor_column") or 1
                        selection_start_line = metadata.get("selection_start_line")
                        selection_start_column = metadata.get("selection_start_column")
                        selection_end_line = metadata.get("selection_end_line")
                        selection_end_column = metadata.get("selection_end_column")

                        truncated = _truncate_file_content(file_content) if file_content else "(empty document)"
                        file_content_with_line_numbers = _add_line_numbers(truncated)
                        selection_ctx = _build_selection_context(
                            selection, cursor_context,
                            selection_start_line, selection_end_line
                        )
                        selection_info = _build_selection_info(
                            selection, selection_start_line, selection_start_column,
                            selection_end_line, selection_end_column
                        )

                        # Get last user message
                        request_text = ""
                        for msg in reversed(messages):
                            if hasattr(msg, "type") and msg.type == "human":
                                request_text = msg.content
                                break
                        if not request_text and messages:
                            request_text = str(messages[-1].content)

                        # Real token-by-token streaming via LLM astream
                        llm = get_llm()
                        chain = LATEX_EDITOR_PROMPT | llm
                        async for chunk in chain.astream({
                            "filename": filename,
                            "file_content_with_line_numbers": file_content_with_line_numbers,
                            "selection_context": selection_ctx,
                            "selection_info": selection_info,
                            "cursor_line": cursor_line,
                            "cursor_column": cursor_column,
                            "request": request_text,
                        }):
                            token = _extract_token(chunk.content) if hasattr(chunk, "content") else ""
                            if token:
                                yield f"data: {token.replace(chr(10), chr(92) + 'n')}\n\n"

                        yield f"data: [META]{json.dumps({'agent': 'latex_editor', 'intent': 'latex_editor', 'sources': []})}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    # Emit early META so FE shows the agent badge
                    try:
                        agent = get_agent(intent)
                        agent_name = agent.name
                    except KeyError:
                        agent_name = intent
                    yield f"data: [META]{json.dumps({'agent': agent_name, 'intent': intent})}\n\n"

            # ── Pipeline complete — stream the answer ─────────────────
            intent = final_state.get("intent", "chat")
            rag_context = final_state.get("rag_context", "")
            web_context = final_state.get("web_context", "")
            sources = final_state.get("sources", [])

            # Resolve agent name for META
            try:
                agent = get_agent(intent)
                agent_name = agent.name
            except KeyError:
                agent_name = intent

            # For RAG+Web combined context
            if intent == "rag" and web_context:
                # RAG was insufficient, web supplemented → use both contexts
                intent = "rag"
                rag_context = rag_context + "\n\n" + web_context if rag_context else web_context

            # Build system prompt with gathered context
            system_prompt = _build_system_prompt(intent, rag_context, web_context, metadata)

            # Stream answer tokens
            llm = get_llm()
            llm_messages = [SystemMessage(content=system_prompt)] + messages

            async for chunk in llm.astream(llm_messages):
                token = _extract_token(chunk.content)
                if token:
                    yield f"data: {token.replace(chr(10), chr(92) + 'n')}\n\n"

            # ── Final META with sources ───────────────────────────────
            yield f"data: [META]{json.dumps({'agent': agent_name, 'intent': intent, 'sources': sources})}\n\n"
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


# ── Non-streaming endpoint (backward compatibility) ──────────────────────────

@router.post("/chat/sync", response_model=ChatResponse)
async def chat_sync(request: ChatRequest):
    """Non-streaming version — returns full response at once via the Supervisor Pipeline.

    Uses the same pipeline as the streaming endpoint so RAG / web-search routing
    is available here too.  The legacy router.get_graph() had no RAG node.
    """
    # Generate trace ID for observability
    trace_id = uuid.uuid4().hex[:8]
    set_trace_id(trace_id)

    messages: list = []
    for msg in request.messages:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))

    # Extract last user message for guard context
    last_user_msg = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            last_user_msg = msg.content
            break
    set_user_message(last_user_msg)

    # Build full PipelineState — same as the streaming endpoint
    # so RAG retrieval and web search work correctly here too.
    state = {
        "messages": messages,
        "document_ids": request.document_ids or [],
        "web_search_enabled": bool(request.web_search_sites),
        "web_search_sites": request.web_search_sites or [],
        "metadata": {
            "project_id": request.project_id,
            "document_ids": request.document_ids,
            "intent_hint": request.intent_hint,
            "web_search_sites": request.web_search_sites,
            "workspace_id": request.workspace_id,
            "user_id": request.user_id,
            "chat_id": request.chat_id,
        },
        "user_id": request.user_id or "",
        "chat_id": request.chat_id or "",
    }

    pipeline = get_pipeline()
    result = await pipeline.ainvoke(state)

    agent_output = result.get("agent_output", {})
    return ChatResponse(
        output=AgentOutput(**agent_output) if agent_output else AgentOutput(
            agent_name=result.get("intent", "chat"),
            content=result.get("early_reply", ""),
            sources=[],
        ),
        intent=result.get("intent"),
    )

