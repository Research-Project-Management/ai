"""
LaTeX Editor Agent — Context-aware AI assistant for the Flux LaTeX editor.
==========================================================================

This agent is invoked when the user interacts with the AI panel or inline
command inside the LaTeX editor. Unlike the generic ``latex`` agent, this one:

- Receives the **full file content** as context
- Receives the **selection / cursor context** (active paragraph or selected text)
- Supports multiple task modes: write, improve, translate, syntax, fix, explain
- Always responds in the same language as the user (Vietnamese or English)
- Wraps any LaTeX output in ```latex blocks for the frontend to detect

Routing:
    The frontend sends ``intent_hint: "latex_editor"`` to bypass Supervisor
    classification and reach this agent directly.

Registration:
    @register_agent → auto-discovered by the Supervisor registry.
    The Supervisor will NOT offer it as a routing option (it's always
    triggered via intent_hint from the editor panel).
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from app.agents.base import AgentState, BaseAgent
from app.agents.registry import register_agent
from app.config import get_llm

logger = logging.getLogger(__name__)

# ── System Prompt ─────────────────────────────────────────────────────────────

LATEX_EDITOR_SYSTEM = """\
You are **Flux AI**, an expert LaTeX writing assistant embedded directly in \
the Flux research editor. You have full access to the document the user is \
currently writing.

## Document Context
**Filename**: {filename}
**Full document content** (for background understanding):
```
{file_content}
```

## Active Context (where the user is working)
{selection_context}

## Your Capabilities

### 1. Academic Writing
- Expand a short idea into a formal academic paragraph
- Improve writing style, grammar, and academic tone
- Continue writing from the cursor position
- Translate text (Vietnamese ↔ English) while preserving LaTeX markup

### 2. LaTeX Syntax Generation
- Generate tables (`tabular`, `booktabs` style)
- Generate equations (`equation`, `align`, `gather` environments)
- Generate figures (`figure`, `includegraphics`, `caption`)
- Generate algorithms (`algorithm`, `algorithmic`)
- Generate BibTeX entries from paper descriptions
- Fix compilation errors (explain the error + provide corrected code)

### 3. Document Analysis
- Summarize a section
- Explain a complex equation or concept
- Suggest improvements for structure or flow

## Output Rules
1. **Language**: Always respond in the **same language** as the user's request
2. **LaTeX code**: ALWAYS wrap in ` ```latex ``` ` blocks — never inline
3. **Prose**: Use clean Markdown for non-LaTeX responses
4. **Completeness**: Provide complete, compilable LaTeX — no placeholders like `% TODO`
5. **Packages**: Use standard packages: `amsmath`, `amssymb`, `graphicx`, `booktabs`, `algorithm2e`
6. **Context-aware**: Reference the document content when relevant (e.g., "Based on your Introduction section...")
7. **Concise**: Be direct — researchers value efficiency
"""

LATEX_EDITOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", LATEX_EDITOR_SYSTEM),
        ("human", "{request}"),
    ]
)


# ── Helper ────────────────────────────────────────────────────────────────────

def _truncate_file_content(content: str, max_chars: int = 8000) -> str:
    """Truncate file content to fit within context window.

    Keeps the beginning and end of the file (most relevant for context).
    Inserts a truncation notice in the middle.
    """
    if len(content) <= max_chars:
        return content

    half = max_chars // 2
    head = content[:half]
    tail = content[-half:]
    omitted = len(content) - max_chars
    return (
        f"{head}\n\n"
        f"... [{omitted} characters omitted — middle of document] ...\n\n"
        f"{tail}"
    )


def _build_selection_context(selection: str | None, cursor_context: str | None) -> str:
    """Build a human-readable selection/cursor context block."""
    parts = []

    if selection and selection.strip():
        parts.append(
            f"**Selected text** (the user has highlighted this):\n"
            f"```\n{selection.strip()}\n```"
        )

    if cursor_context and cursor_context.strip():
        label = "**Surrounding context** (text around the cursor):" if not selection else "**Surrounding paragraph**:"
        parts.append(f"{label}\n```\n{cursor_context.strip()}\n```")

    if not parts:
        return "*(No specific selection — responding to a general question about the document)*"

    return "\n\n".join(parts)


# ── Agent ─────────────────────────────────────────────────────────────────────

@register_agent
class LaTeXEditorAgent(BaseAgent):
    """Context-aware AI assistant for the Flux LaTeX editor.

    Handles all AI interactions originating from the editor panel
    (chat panel + inline commands). Always receives the full document
    as context.

    Triggered via ``intent_hint: "latex_editor"`` from the frontend.
    """

    name = "latex_editor"
    description = (
        "Context-aware LaTeX writing assistant for the editor panel. "
        "Use ONLY when triggered by the LaTeX editor with intent_hint='latex_editor'. "
        "Handles: academic writing, LaTeX syntax generation, equation help, "
        "table generation, error fixing, and document-aware Q&A. "
        "Has access to the full document content and active selection."
    )

    async def run(self, state: AgentState) -> AgentState:
        llm = get_llm()

        messages = state.get("messages", [])
        metadata = state.get("metadata", {})

        # Extract the user's request
        request = ""
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                request = msg.content
                break
        if not request and messages:
            request = str(messages[-1].content)

        # Extract editor context from metadata
        file_content = metadata.get("file_content", "")
        filename = metadata.get("filename", "document.tex")
        selection = metadata.get("selection", "")
        cursor_context = metadata.get("cursor_context", "")

        # Build context blocks
        truncated_content = _truncate_file_content(file_content) if file_content else "(empty document)"
        selection_ctx = _build_selection_context(selection, cursor_context)

        logger.info(
            "LaTeXEditor: file=%s len=%d selection=%d cursor=%d",
            filename,
            len(file_content),
            len(selection or ""),
            len(cursor_context or ""),
        )

        # Invoke the chain
        chain = LATEX_EDITOR_PROMPT | llm
        response = await chain.ainvoke(
            {
                "filename": filename,
                "file_content": truncated_content,
                "selection_context": selection_ctx,
                "request": request,
            }
        )

        content = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

        state["agent_output"] = {
            "agent_name": self.name,
            "content": content,
            "sources": [],
            "metadata": {
                "type": "latex_editor",
                "filename": filename,
                "has_selection": bool(selection),
            },
        }
        state["messages"] = [AIMessage(content=content)]
        return state
