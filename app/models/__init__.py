"""Pydantic request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Request Models ───────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    """Unified chat request — the router decides which agent handles it."""

    messages: list[ChatMessage]
    project_id: str | None = None
    document_ids: list[str] | None = None
    intent_hint: str | None = Field(
        None,
        description="Optional hint to force a specific agent: chat, rag, analyze, latex, task, web_search, action",
    )
    web_search_sites: list[str] | None = Field(
        None,
        description="Sites to restrict web search to. Defaults to curated academic sources when web_search intent is used.",
    )
    # Agent context — injected by RPM-BE proxy for action agent
    workspace_id: str | None = Field(
        None,
        description="Workspace ID or URL slug. Required for action agent operations.",
    )
    user_id: str | None = Field(
        None,
        description="Authenticated user's MongoDB ObjectId. Injected by RPM-BE proxy.",
    )
    chat_id: str | None = Field(
        None,
        description="Chat session ID. Documents are scoped to this session — used for RAG isolation.",
    )
    # ── LaTeX Editor context (only sent when intent_hint='latex_editor') ──
    file_content: str | None = Field(
        None,
        description="Full content of the LaTeX file currently open in the editor.",
    )
    filename: str | None = Field(
        None,
        description="Filename of the active LaTeX file (e.g. 'main.tex').",
    )
    selection: str | None = Field(
        None,
        description="Text currently selected by the user in the editor.",
    )
    cursor_context: str | None = Field(
        None,
        description="Surrounding paragraph / lines around the cursor position.",
    )
    # ── Cursor & selection range (v2 rich context) ────────────────────────────
    cursor_line: int | None = Field(
        None,
        description="1-based cursor line number in the editor.",
    )
    cursor_column: int | None = Field(
        None,
        description="1-based cursor column number in the editor.",
    )
    selection_start_line: int | None = Field(
        None,
        description="1-based start line of user selection.",
    )
    selection_start_column: int | None = Field(
        None,
        description="1-based start column of user selection.",
    )
    selection_end_line: int | None = Field(
        None,
        description="1-based end line of user selection.",
    )
    selection_end_column: int | None = Field(
        None,
        description="1-based end column of user selection.",
    )
    document_structure: str | None = Field(
        None,
        description="Summary of document LaTeX structure (sections, packages, labels).",
    )
    command_hint: str | None = Field(
        None,
        description="Slash command hint, e.g. 'fix', 'refactor', 'complete'.",
    )
    memory_context: dict[str, Any] | None = Field(
        None,
        description="Persistent chat/workspace/project memory injected by RPM-BE.",
    )


class DocumentUploadMeta(BaseModel):
    title: str | None = None
    tags: list[str] = Field(default_factory=list)


# ── Response Models ──────────────────────────────────────────────

class AgentOutput(BaseModel):
    """Structured output from any agent."""

    agent_name: str
    content: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Non-streaming response wrapper."""

    output: AgentOutput
    intent: str | None = None


class DocumentInfo(BaseModel):
    id: str
    title: str
    chunk_count: int
    tags: list[str] = Field(default_factory=list)


class HealthStatus(BaseModel):
    status: str = "ok"
    services: dict[str, str] = Field(default_factory=dict)
