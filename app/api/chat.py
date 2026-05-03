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
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.graph import PipelineState, get_pipeline
from app.agents.orchestrator import stream_orchestrator
from app.agents.registry import get_agent
from app.config import get_agent_llm, get_llm
from app.models import AgentOutput, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


# ── Token extractor ──────────────────────────────────────────────

def _extract_token(content) -> str:
    """Safely extract text from a LangChain chunk's content.

    Handles:
    - ``str`` — returned as-is
    - ``list`` of content blocks (Anthropic / Groq structured format):
      e.g. [{'type': 'text', 'text': 'Hello'}, {'type': 'text', 'text': ' world'}]
    - Anything else — str() conversion (should not normally happen)
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                # ignore non-text block types (e.g. tool_use, thinking)
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    # fallback — avoid Python repr of complex objects
    return ""


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


# ── Action Agent Streaming ───────────────────────────────────────────────────

async def _stream_action_agent(messages: list, request: ChatRequest):
    """Run the ReAct action agent and yield SSE events for tool calls + tokens.

    This uses LangGraph's create_react_agent with astream_events to capture
    both tool invocations and LLM response tokens in real-time.
    """
    from langgraph.prebuilt import create_react_agent

    from app.agents.action_agent import CONFIRM_REQUIRED_TOOLS, build_action_prompt
    from app.tools import ALL_TOOLS
    from app.tools._client import set_user_context

    # Set user context for tool calls
    user_id = request.user_id or ""
    workspace_id = request.workspace_id or ""
    project_id = request.project_id or ""
    set_user_context(user_id, workspace_id, project_id)

    # Build the action agent
    system_prompt = build_action_prompt(
        workspace_id=workspace_id,
        project_id=project_id,
        user_id=user_id,
    )

    agent_llm = get_agent_llm()
    agent = create_react_agent(
        model=agent_llm,
        tools=ALL_TOOLS,
        prompt=system_prompt,
    )

    # Track tool calls for summary
    tool_calls_log: list[dict] = []

    # Stream agent events
    async for event in agent.astream_events(
        {"messages": messages},
        version="v2",
    ):
        kind = event.get("event", "")

        # ── Tool call start ──────────────────────────────────────
        if kind == "on_tool_start":
            tool_name = event.get("name", "unknown")
            tool_input = event.get("data", {}).get("input", {})

            # Check if this tool requires confirmation
            needs_confirm = tool_name in CONFIRM_REQUIRED_TOOLS

            action_event = {
                "type": "tool_start",
                "tool": tool_name,
                "input": tool_input,
                "needs_confirm": needs_confirm,
            }
            yield f"data: [ACTION]{json.dumps(action_event)}\n\n"
            logger.info("Tool start: %s(%s)", tool_name, json.dumps(tool_input)[:200])

        # ── Tool call end ────────────────────────────────────────
        elif kind == "on_tool_end":
            tool_name = event.get("name", "unknown")
            tool_output = event.get("data", {}).get("output", "")

            # Parse output — could be string or ToolMessage
            if hasattr(tool_output, "content"):
                output_data = tool_output.content
            else:
                output_data = str(tool_output)

            # Try to parse as JSON for structured display
            try:
                if isinstance(output_data, str):
                    parsed_output = json.loads(output_data)
                else:
                    parsed_output = output_data
            except (json.JSONDecodeError, TypeError):
                parsed_output = {"raw": output_data[:500] if isinstance(output_data, str) else str(output_data)[:500]}

            action_event = {
                "type": "tool_end",
                "tool": tool_name,
                "output": parsed_output,
            }
            yield f"data: [ACTION]{json.dumps(action_event, default=str)}\n\n"

            tool_calls_log.append({
                "tool": tool_name,
                "success": not (isinstance(parsed_output, dict) and parsed_output.get("error")),
            })
            logger.info("Tool end: %s → success=%s", tool_name, tool_calls_log[-1]["success"])

        # ── LLM answer tokens (final response) ──────────────────
        elif kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                token = _extract_token(chunk.content)
                if token:
                    yield f"data: {token.replace(chr(10), chr(92) + 'n')}\n\n"


# ── Streaming Endpoint ───────────────────────────────────────────────────────

@router.post("/chat")
async def chat(request: ChatRequest):
    """Unified chat endpoint with Supervisor Pipeline + real token streaming."""

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
                        yield f"data: [META]{json.dumps({'agent': 'orchestrator', 'intent': 'action'})}\n\n"

                        # Stream qua A2A Orchestrator (delegate sang sub-agents)
                        async for event_data in stream_orchestrator(messages, request):
                            yield event_data

                        # Final META
                        yield f"data: [META]{json.dumps({'agent': 'orchestrator', 'intent': 'action', 'sources': []})}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    # ── LATEX EDITOR AGENT ROUTE ─────────────────────────────
                    # Builds its own context-aware system prompt using file_content,
                    # selection, and cursor_context from the request metadata.
                    if route == "latex_editor":
                        from app.agents.latex_editor_agent import LaTeXEditorAgent
                        yield f"data: [META]{json.dumps({'agent': 'latex_editor', 'intent': 'latex_editor'})}\n\n"

                        editor_agent = LaTeXEditorAgent()
                        agent_state: dict = {
                            "messages": messages,
                            "metadata": metadata,
                        }
                        result_state = await editor_agent.run(agent_state)
                        content = result_state.get("agent_output", {}).get("content", "")

                        # Stream the content token by token (word-level chunking)
                        words = content.split(" ")
                        for i, word in enumerate(words):
                            chunk = word + (" " if i < len(words) - 1 else "")
                            if chunk:
                                yield f"data: {chunk.replace(chr(10), chr(92) + 'n')}\n\n"

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
            combined_web = web_context
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
