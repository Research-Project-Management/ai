"""Persistent memory summarization endpoints.

RPM-BE owns MongoDB persistence. This service only distills a chat window into
structured memory, then RPM-BE stores the result without blocking the user chat.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import get_router_llm

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryUpdateMessage(BaseModel):
    role: str
    content: str


class MemoryUpdateRequest(BaseModel):
    chat_id: str
    user_id: str
    workspace_id: str
    project_id: str | None = None
    existing_summary: str = ""
    existing_key_facts: list[str] = Field(default_factory=list)
    existing_open_questions: list[str] = Field(default_factory=list)
    messages: list[MemoryUpdateMessage]


class MemoryItem(BaseModel):
    scope: str = "workspace"
    type: str = "workspace_summary"
    content: str
    confidence: float = 0.7


class MemoryUpdateResponse(BaseModel):
    summary: str = ""
    key_facts: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    memories: list[MemoryItem] = Field(default_factory=list)


def _extract_json(text: str) -> dict[str, Any]:
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end != -1 and end > start:
        clean = clean[start : end + 1]
    return json.loads(clean)


def _clip_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:limit]


@router.post("/update", response_model=MemoryUpdateResponse)
async def update_memory(request: MemoryUpdateRequest):
    """Summarize recent chat into durable chat and workspace/project memory."""
    transcript = "\n".join(
        f"{msg.role}: {msg.content[:1800]}" for msg in request.messages[-30:]
    )
    project_scope_note = (
        "Use project scope for memories that are specifically about the active project."
        if request.project_id
        else "No active project is selected; use workspace scope."
    )

    prompt = (
        "You maintain long-term memory for Flux, a research workspace AI.\n"
        "Update memory from the recent transcript. Keep durable facts only: user "
        "preferences, project/workspace goals, decisions, constraints, named entities, "
        "and unresolved follow-ups. Do not store temporary small talk.\n\n"
        f"Workspace ID: {request.workspace_id}\n"
        f"Project ID: {request.project_id or 'none'}\n"
        f"{project_scope_note}\n\n"
        "Existing chat summary:\n"
        f"{request.existing_summary or '(none)'}\n\n"
        "Existing key facts:\n"
        + "\n".join(f"- {fact}" for fact in request.existing_key_facts[:12])
        + "\n\nExisting open questions:\n"
        + "\n".join(f"- {question}" for question in request.existing_open_questions[:8])
        + "\n\nRecent transcript:\n"
        + transcript
        + "\n\nReturn ONLY JSON with this shape:\n"
        "{\n"
        '  "summary": "compact running summary, max 1200 chars",\n'
        '  "key_facts": ["max 16 short facts"],\n'
        '  "open_questions": ["max 10 unresolved questions"],\n'
        '  "memories": [\n'
        '    {"scope": "workspace|project", "type": "project_summary|workspace_summary|preference|decision|entity|constraint", "content": "durable memory", "confidence": 0.0}\n'
        "  ]\n"
        "}"
    )

    try:
        llm = get_router_llm()
        response = await llm.ainvoke(
            [
                {
                    "role": "system",
                    "content": "You are a precise JSON-only memory distiller.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        content = response.content if isinstance(response.content, str) else str(response.content)
        parsed = _extract_json(content)
    except Exception as exc:
        logger.warning("Memory update failed, falling back to existing memory: %s", exc)
        return MemoryUpdateResponse(
            summary=request.existing_summary,
            key_facts=request.existing_key_facts[:16],
            open_questions=request.existing_open_questions[:10],
            memories=[],
        )

    memories = []
    for item in parsed.get("memories", []) if isinstance(parsed.get("memories"), list) else []:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        scope = item.get("scope") if item.get("scope") in {"workspace", "project"} else "workspace"
        mem_type = item.get("type")
        if mem_type not in {
            "project_summary",
            "workspace_summary",
            "preference",
            "decision",
            "entity",
            "constraint",
        }:
            mem_type = "project_summary" if scope == "project" else "workspace_summary"
        try:
            confidence = float(item.get("confidence", 0.7))
        except Exception:
            confidence = 0.7
        memories.append(
            MemoryItem(
                scope=scope,
                type=mem_type,
                content=content[:500],
                confidence=max(0, min(1, confidence)),
            )
        )

    return MemoryUpdateResponse(
        summary=str(parsed.get("summary") or request.existing_summary)[:1200],
        key_facts=_clip_list(parsed.get("key_facts"), 16),
        open_questions=_clip_list(parsed.get("open_questions"), 10),
        memories=memories[:8],
    )
