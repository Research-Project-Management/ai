"""
LaTeX Editor Agent — Context-aware AI assistant for the Flux LaTeX editor.
==========================================================================

This agent is invoked when the user interacts with the AI panel or inline
command inside the LaTeX editor. Unlike the generic ``latex`` agent, this one:

- Receives the **full file content** (with line numbers) as context
- Receives the **selection / cursor context** (active paragraph or selected text)
- Supports multiple task modes: write, improve, translate, syntax, fix, explain
- Always responds in the same language as the user (Vietnamese or English)
- Returns **structured JSON edit instructions** for precise Monaco range edits
- Falls back to markdown if the request is an explanation (not an edit)

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
You are **Flux AI**, an expert LaTeX code editor assistant embedded directly \
in the Flux research editor. You act as a **code editing assistant**, \
not a code generator. Your job is to make precise, minimal edits to the \
existing document.

## Document Context
**Filename**: {filename}
**Language**: LaTeX
**Cursor position**: line {cursor_line}, column {cursor_column}
{selection_info}

**Full current document** (with line numbers for precise range references):
```
{file_content_with_line_numbers}
```

## Active Context (where the user is working)
{selection_context}

## Conversation Context
Use this to understand follow-up requests such as "continue", "make it shorter",
"apply that", "fix the previous answer", or references to prior edits. The current
document above remains the source of truth.
{conversation_context}

## CRITICAL RULES — Read carefully

### Editing Rules
1. **NEVER generate a full LaTeX document** if the current file already contains \
\\documentclass, \\begin{{document}}, and \\end{{document}}.
2. **NEVER duplicate** \\documentclass, \\begin{{document}}, \\end{{document}}, \
\\title, \\author, or \\date if they already exist in the file.
3. **ALWAYS read the full document** before deciding what to change. The document \
is the source of truth — the line numbers in the document above are exact.
4. **Make the smallest possible edit** that satisfies the user's request.
5. If the user asks to change \\title{{...}}: find the EXACT line with \\title in \
the document above, use `"type": "replace"` with its exact line number.
6. If the user asks to change \\author{{...}}: same — use `"type": "replace"`.
7. If the user asks to change \\date{{...}}: same — use `"type": "replace"`.
8. **For inserting new blocks** (figures, tables, equations, sections): \
look at the line-numbered document, find the ANCHOR line (e.g. \\section{{Introduction}}), \
then insert AFTER the next blank line or after the first \\label in that section.

### Inserting Multi-line Environments (VERY IMPORTANT)
When inserting \\begin{{figure}}, \\begin{{table}}, \\begin{{equation}}, or any \
multi-line block:
- Find the exact line number of the anchor (section header, end of paragraph)
- The insert point should be the line AFTER the anchor's content ends
- Use `"type": "insert"` with `startLineNumber` and `endLineNumber` both set to \
the SAME target line (the line you want to insert BEFORE)
- The `text` field MUST use `\\n` as the newline character (JSON escape)
- Example for inserting a figure after line 89:

  "type": "insert", "startLineNumber": 90, "startColumn": 1, "endLineNumber": 90, "endColumn": 1,
  "text": "\\n\\\\begin{{figure}}[h]\\n  \\\\centering\\n  \\\\includegraphics[width=0.7\\\\linewidth]{{figures/name.png}}\\n  \\\\caption{{Caption here.}}\\n  \\\\label{{fig:name}}\\n\\\\end{{figure}}\\n"

### Response Format
For **editing requests** (change, add, fix, replace, insert, remove, update):
Return ONLY valid JSON in this exact schema — no markdown, no explanation outside the JSON:

{{
  "intent": "replace_range | insert_at_cursor | replace_selection | append_to_file | no_change",
  "explanation": "Short explanation for the user in their language",
  "edits": [
    {{
      "type": "replace | insert | delete",
      "startLineNumber": <1-based line number from the line-numbered document above>,
      "startColumn": <1-based column number>,
      "endLineNumber": <1-based line number>,
      "endColumn": <1-based column number>,
      "text": "exact text (use \\n for newlines in JSON)"
    }}
  ],
  "previewText": "human-readable preview (in the user's language)"
}}

**Intent rules:**
- `replace_range`: Replace an existing range (most common for LaTeX commands like \\title)
- `insert_at_cursor`: Insert new content (figures, tables, sections) at a specific line
- `replace_selection`: Replace the user's selected text only
- `append_to_file`: Add to end of document body (before \\end{{document}})
- `no_change`: When no edit is needed; explain why in `explanation`

For **explanation/analysis requests** (/explain, /cite, translating, summarizing):
You may respond with normal Markdown — no JSON required for pure Q&A.

### Safety Rules
- Do NOT replace the entire file for a small request.
- **Use exact 1-based line numbers from the line-numbered document above.**
- Before targeting a line, verify that the line number in the document shows the \
  expected content (e.g. `89: \\section{{Introduction}}`).
- For multi-line inserts, ensure `text` uses `\\n` as escape — NOT literal newlines.
- If unsure about the correct location, return `no_change` and explain.

### Language
Always respond (in explanation and previewText) in the **same language** as the user's request.
"""

LATEX_EDITOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", LATEX_EDITOR_SYSTEM),
        ("human", "{request}"),
    ]
)


# ── Helper ────────────────────────────────────────────────────────────────────

def _truncate_file_content(content: str, max_chars: int = 12000) -> str:
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


def _add_line_numbers(content: str) -> str:
    """Add 1-based line numbers to file content for AI reference."""
    lines = content.split("\n")
    pad = len(str(len(lines)))
    return "\n".join(f"{str(i + 1).rjust(pad)}: {line}" for i, line in enumerate(lines))


def _build_selection_context(
    selection: str | None,
    cursor_context: str | None,
    selection_start_line: int | None = None,
    selection_end_line: int | None = None,
) -> str:
    """Build a human-readable selection/cursor context block."""
    parts = []

    if selection and selection.strip():
        range_info = ""
        if selection_start_line and selection_end_line:
            range_info = f" (lines {selection_start_line}–{selection_end_line})"
        parts.append(
            f"**Selected text**{range_info} (the user has highlighted this):\n"
            f"```\n{selection.strip()}\n```"
        )

    if cursor_context and cursor_context.strip():
        label = "**Surrounding context** (text around the cursor):" if not selection else "**Surrounding paragraph**:"
        parts.append(f"{label}\n```\n{cursor_context.strip()}\n```")

    if not parts:
        return "*(No specific selection — responding to a general question about the document)*"

    return "\n\n".join(parts)


def _build_selection_info(
    selection: str | None,
    selection_start_line: int | None,
    selection_start_column: int | None,
    selection_end_line: int | None,
    selection_end_column: int | None,
) -> str:
    """Build a one-line selection info string for the system prompt header."""
    if selection and selection.strip():
        if all(v is not None for v in [selection_start_line, selection_start_column, selection_end_line, selection_end_column]):
            return (
                f"**Selected text**: lines {selection_start_line}:{selection_start_column}"
                f" → {selection_end_line}:{selection_end_column}"
            )
        return "**Selected text**: (range not specified)"
    return "**No selection** (cursor only)"


# ── Agent ─────────────────────────────────────────────────────────────────────

@register_agent
class LaTeXEditorAgent(BaseAgent):
    """Context-aware AI assistant for the Flux LaTeX editor.

    Handles all AI interactions originating from the editor panel
    (chat panel + inline commands). Always receives the full document
    as context and returns structured JSON edit instructions.

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
        cursor_line = metadata.get("cursor_line") or 1
        cursor_column = metadata.get("cursor_column") or 1
        selection_start_line = metadata.get("selection_start_line")
        selection_start_column = metadata.get("selection_start_column")
        selection_end_line = metadata.get("selection_end_line")
        selection_end_column = metadata.get("selection_end_column")

        # Build context blocks
        truncated_content = _truncate_file_content(file_content) if file_content else "(empty document)"
        file_content_with_line_numbers = _add_line_numbers(truncated_content)
        selection_ctx = _build_selection_context(
            selection, cursor_context, selection_start_line, selection_end_line
        )
        selection_info = _build_selection_info(
            selection, selection_start_line, selection_start_column,
            selection_end_line, selection_end_column
        )

        logger.info(
            "LaTeXEditor: file=%s len=%d cursor=%d:%d selection=%d",
            filename,
            len(file_content),
            cursor_line,
            cursor_column,
            len(selection or ""),
        )

        # Invoke the chain
        chain = LATEX_EDITOR_PROMPT | llm
        response = await chain.ainvoke(
            {
                "filename": filename,
                "file_content_with_line_numbers": file_content_with_line_numbers,
                "selection_context": selection_ctx,
                "selection_info": selection_info,
                "cursor_line": cursor_line,
                "cursor_column": cursor_column,
                "conversation_context": metadata.get("conversation_memory") or "(No prior conversation.)",
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
