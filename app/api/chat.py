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
from app.agents.conversation_context import (
    compact_messages_for_llm,
    memory_system_message,
)
from app.agents.orchestrator import stream_orchestrator
from app.agents.registry import get_agent
from app.config import get_agent_llm, get_llm
from app.models import AgentOutput, ChatRequest, ChatResponse
from app.tools._client import set_trace_id, get_trace_id, set_user_message

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


from app.agents.utils import _extract_token


RESPONSE_QUALITY_GUIDE = (
    "## RESPONSE QUALITY & COGNITIVE DEPTH GUIDELINE\n"
    "- **Intellectual Rigor & Authority**: Act as an elite principal scientist, senior academic reviewer, and world-class researcher. Your answers must be intellectually rich, precise, and scientifically sound. Avoid superficial explanations.\n"
    "- **Multi-Dimensional Analysis**: Do not just give direct answers; explain the underlying theoretical frameworks, mechanisms, trade-offs, and design choices. Present multiple perspectives or alternative hypotheses when relevant.\n"
    "- **Structural Sophistication**: Structure responses systematically. Use rich typography, bolding for critical terms, clean nested bullet points, and markdown tables for comparative analysis, architecture summaries, or parameter trade-offs.\n"
    "- **Language & Tone Matching**: Speak in a sophisticated, professional, and academically elegant tone. Adapt seamlessly to the user's technical level. If they write in Vietnamese, respond in refined, professional, and natural academic Vietnamese.\n"
    "- **Exhaustiveness Over Curtness**: Never output generic, one-sentence summaries when a comprehensive explanation adds value. Explain 'why' and 'how', not just 'what'. Provide concrete examples, code/LaTeX snippets, and solid scientific reasoning.\n"
    "- **Honest Epistemology**: Clearly state any assumptions made, boundary conditions, or limitations when the available context is incomplete or ambiguous. Never guess or hallucinate.\n"
)


def _format_persistent_memory(memory_context: dict | None) -> str:
    """Format persisted chat/workspace/project memory for prompt injection."""
    if not memory_context:
        return ""

    sections: list[str] = []
    chat_summary = str(memory_context.get("chatSummary") or "").strip()
    if chat_summary:
        sections.append("Chat summary:\n" + chat_summary)

    key_facts = [
        str(item).strip()
        for item in memory_context.get("keyFacts") or []
        if str(item).strip()
    ][:12]
    if key_facts:
        sections.append("Key facts:\n" + "\n".join(f"- {item}" for item in key_facts))

    open_questions = [
        str(item).strip()
        for item in memory_context.get("openQuestions") or []
        if str(item).strip()
    ][:8]
    if open_questions:
        sections.append(
            "Open questions:\n" + "\n".join(f"- {item}" for item in open_questions)
        )

    memory_lines: list[str] = []
    for memory in (memory_context.get("memories") or [])[:12]:
        if not isinstance(memory, dict):
            continue
        content = str(memory.get("content") or "").strip()
        if not content:
            continue
        label = str(memory.get("type") or "memory").replace("_", " ")
        scope = str(memory.get("scope") or "workspace")
        memory_lines.append(f"- [{scope} / {label}] {content}")
    if memory_lines:
        sections.append("Workspace/project memory:\n" + "\n".join(memory_lines))

    return "\n\n".join(sections).strip()


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
        selected_excerpt = str(metadata.get("selection") or "").strip()
        selection_block = (
            "\n\n## User-Selected Passage\n"
            "The user selected this passage in the reader. Use it as the focus of the answer, "
            "but keep the answer grounded in the source passages.\n"
            + selected_excerpt[:4000]
            if selected_excerpt
            else ""
        )
        return (
            "You are an elite Senior Principal Scientist specializing in deep document synthesis, "
            "literature review, and cross-reference analysis. Your goal is to answer the user's "
            "questions with maximum technical depth, relying STRICTLY on the uploaded source "
            "documents provided below.\n\n"
            + RESPONSE_QUALITY_GUIDE
            + "\n"
            "## RIGOROUS RULES FOR SYNTHESIS\n"
            "1. **Absolute Faithfulness**: Rely ONLY on facts directly stated or logically inferred in the Source Passages. Never invent information or use general knowledge beyond the sources.\n"
            "2. **Precise Multi-Source Cross-Referencing**: When synthesizing, compare findings across different files. Point out consensus, differences in methodology, or conflicting results between authors.\n"
            "3. **Professional Academic Citations**: Cite facts inline precisely using the format **[Source: filename]** (e.g. `[Source: deep_learning_paper.pdf]`). Do not cite if the source does not explicitly support the claim.\n"
            "4. **Comprehensive Structure**: For summaries or comparisons, construct an analytical outline containing an executive summary, methodology comparison, core findings, and open questions/limitations.\n"
            "5. **Honesty & Epistemic Humility**: If the sources do not contain the answer, state: 'The provided source documents do not contain sufficient evidence to answer this question.' and suggest what relevant information is missing.\n"
            "6. **Rich Formatting**: Use professional markdown elements. Use tables to compare variables, bolding for key parameters, and clean hierarchy.\n\n"
            "## SOURCE PASSAGES\n" + ctx_text + selection_block
        )

    if intent == "web_search":
        ctx_text = web_context or "(No search results found.)"
        return (
            "You are a distinguished Research Librarian and Scientific Intelligence Expert. "
            "The user asked a technical question, and real-time search results are provided below. "
            "Your task is to synthesize these findings into a state-of-the-art literature review.\n\n"
            + RESPONSE_QUALITY_GUIDE
            + "\n"
            "## LIT-REVIEW RULES\n"
            "1. **Deep Synthesis**: Do not list results title-by-title. Instead, group findings by core themes, methodologies, or historical progress. Synthesize what the collective data means.\n"
            "2. **Critical Evaluation**: Note when multiple sources agree on a breakthrough, and call out areas of dispute or ongoing research.\n"
            "3. **Strict Citation Mapping**: Cite facts inline using markdown links `[Title](url)`. Only use exact URLs that appear verbatim in the search results below.\n"
            "4. **Systematic References Section**: End with a structured '## References' section listing all cited articles with clean titles and URLs in bullet format.\n"
            "5. **Professional Terminology**: Use precise scientific terminology. Explain complex technical terms in-context when appropriate.\n\n"
            "## SEARCH RESULTS\n" + ctx_text
        )

    if intent == "analyze":
        return (
            "You are a Senior Peer Reviewer, Data Analyst, and Scientific Auditor. "
            "Provide a highly structured, critical analysis of the data or methodology below:\n\n"
            "## STRUCTURE REQUIREMENTS\n"
            "1. **Executive Assessment**: A high-level, mathematically sound summary of the core results.\n"
            "2. **Methodological Audit**: Analyze the rigor of the study design, sample sizes, control groups, and potential sources of bias or systematic error.\n"
            "3. **Critical Strengths & Limitations**: Evaluate where the findings are robust and where they lack statistical power or external validity.\n"
            "4. **Comparative Synthesis**: Use a Markdown Table comparing the key metrics, parameters, or outcomes discussed in the text.\n"
            "5. **Strategic Implications**: Detail exactly what these findings imply for future research directions, engineering workflows, or practical applications.\n\n"
            + RESPONSE_QUALITY_GUIDE
            + (("\n\n## Content for Analysis\n" + rag_context) if rag_context else "")
        )

    if intent == "latex":
        return (
            "You are a World-Class LaTeX Typographer, Document Architect, and Academically Published Editor. "
            "Generate highly optimized, elegant, and perfectly compilable LaTeX code.\n\n"
            "## TYPOGRAPHY & CODE STANDARDS\n"
            "1. **Wrap Code**: Always wrap your output in standard ```latex blocks.\n"
            "2. **Professional Packages**: Use advanced packages for professional layouts: `amsmath`, `amssymb` (math packages), `graphicx` (figures), `booktabs` (professional tables without vertical borders), `microtype` (typographical spacing), and `hyperref` (links).\n"
            "3. **Perfect Math Typo**: Follow rigorous math spacing rules (e.g. use `\\mathrm{d}` for derivatives, `\\cdot` for multiplication, align equations properly using `align*` environment instead of `$$`).\n"
            "4. **Informative Comments**: Add high-value inline comments explaining complex constructs, macro definitions, or layout decisions.\n"
            "5. **Strict Compilability**: Ensure all braces match, packages are loaded, and the generated code is completely standalone and ready to compile without warning."
        )

    if intent == "latex_editor":
        return (
            "You are a world-class Academic Peer Co-Writer and LaTeX Editor Assistant. "
            "Your role is to help the user co-author, refine, and edit their research paper.\n\n"
            "## EDITING PROTOCOL\n"
            "1. **Academic Prose Refinement**: Improve clarity, flow, academic elegance, and vocabulary transitions. Make the prose sound like it was published in a top-tier journal (Nature, IEEE, ACM).\n"
            "2. **Code Integrity**: Ensure any LaTeX environments are opened and closed perfectly. Never break the compilability of the surrounding document.\n"
            "3. **Markdown Explanations**: When explaining edits, explain the stylistic or structural reasoning behind your choices using clean Markdown.\n"
            "4. **Exact Code Fences**: Wrap code suggestions in standard ```latex blocks."
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
            "You are a Lead Systems Architect, Agile Research Manager, and Project Coordinator. "
            "Formulate a deeply structured, realistic, and highly actionable research roadmap:\n\n"
            "## ROADMAP REQUIREMENTS\n"
            "1. **Task Details**: For each task, specify: Title, Priority (High/Medium/Low), Estimated Time, and Target Deliverables.\n"
            "2. **Technical Details**: Provide a 2-3 sentence technical description of 'how' the task should be implemented (tools, libraries, algorithms).\n"
            "3. **Dependency Mapping**: Explicitly map dependencies and explain why the proposed sequencing is logical to prevent blockers.\n"
            "4. **Risk Assessment**: Identify potential engineering risks, data bottlenecks, or theoretical challenges, and suggest solid mitigation strategies.\n"
            "5. **Actionable Roadmap**: Format cleanly using a nested markdown checklist and comparison tables for timelines.\n\n"
            + RESPONSE_QUALITY_GUIDE
            + task_ctx
        )

    # Default: "chat"
    return (
        "You are **Flux AI**, an elite Scientific Thought Partner and Research Advisor specializing "
        "in academic project management, research design, and computer-aided co-writing. "
        "Your answers should be highly analytical, comprehensive, and intellectually stimulating.\n\n"
        + RESPONSE_QUALITY_GUIDE
        + "\n"
        "## COGNITIVE STYLE\n"
        "- **Deep Explanations**: When answering, go beyond simple factual recall. Explain 'why' a concept is true, 'how' it connects to other fields, and its practical implications.\n"
        "- **Systematic Structuring**: Break down complex problems into analytical sub-components. Use bold terms, mathematical representations where helpful, and clean tables.\n"
        "- **Academic Humility**: Be authoritative yet objective. Clarify boundary conditions, assume nothing, and evaluate limitations."
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

    compact_messages, conversation_memory = compact_messages_for_llm(messages)
    persistent_memory = _format_persistent_memory(request.memory_context)
    if persistent_memory:
        conversation_memory = (
            "Persistent memory:\n"
            + persistent_memory
            + ("\n\nConversation memory:\n" + conversation_memory if conversation_memory else "")
        )
    memory_message = memory_system_message(conversation_memory)

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
        "conversation_memory": conversation_memory,
        "persistent_memory": persistent_memory,
    }

    # Build initial pipeline state
    initial_state: PipelineState = {
        "messages": compact_messages,
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
                        async for event_data in stream_orchestrator(
                            compact_messages,
                            request,
                            conversation_memory,
                        ):
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
                            stream_latex_editor,
                            LATEX_EDITOR_SYSTEM,
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

                        # Formulate the system prompt with full Monaco/LaTeX context
                        system_prompt = LATEX_EDITOR_SYSTEM.format(
                            filename=filename,
                            file_content_with_line_numbers=file_content_with_line_numbers,
                            selection_context=selection_ctx,
                            selection_info=selection_info,
                            cursor_line=cursor_line,
                            cursor_column=cursor_column,
                            conversation_context=conversation_memory or "(No prior conversation.)",
                        )

                        # Stream via the coordinator ReAct agent graph (supports search tools)
                        async for chunk in stream_latex_editor(
                            messages=messages,
                            system_prompt=system_prompt,
                            document_ids=request.document_ids or [],
                            user_id=request.user_id or "",
                            chat_id=request.chat_id or "",
                        ):
                            yield chunk

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
            llm_messages = [SystemMessage(content=system_prompt)]
            if memory_message:
                llm_messages.append(memory_message)
            llm_messages += compact_messages

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

    compact_messages, conversation_memory = compact_messages_for_llm(messages)
    persistent_memory = _format_persistent_memory(request.memory_context)
    if persistent_memory:
        conversation_memory = (
            "Persistent memory:\n"
            + persistent_memory
            + ("\n\nConversation memory:\n" + conversation_memory if conversation_memory else "")
        )

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
        "messages": compact_messages,
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
            "conversation_memory": conversation_memory,
            "persistent_memory": persistent_memory,
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
