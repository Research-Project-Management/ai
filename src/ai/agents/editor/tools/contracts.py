"""Command, Runtime, and Registry contracts for `editor_agent` tools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ai.agents.editor.gateway.contracts import EditorGateway, EditorScope
from ai.agents.editor.patching.contracts import EditorPatchProposal
from ai.orchestrator.contracts import CitationItem
from ai.shared.workflow.registry import ToolRegistry

EditorRiskTier = Literal["read", "propose", "write"]


class CachedFile(BaseModel):
    """In-turn cached page file keyed cleanly without duplicate HTTP reads."""

    model_config = ConfigDict(frozen=True)

    file_id: str | None = None
    file_path: str
    content: str = ""


def _normalize_path(path: str | None) -> str:
    return (path or "").strip().lstrip("/")


@dataclass
class EditorToolRuntime:
    """Mutable per-turn execution context passed to Editor tools (and synced back to State)."""

    gateway: EditorGateway
    scope: EditorScope
    file_cache: dict[str, CachedFile] = field(default_factory=dict)
    known_cite_keys: set[str] = field(default_factory=set)
    known_labels: set[str] = field(default_factory=set)
    patches: list[EditorPatchProposal] = field(default_factory=list)
    max_patches: int = 8

    def get_cached_file(
        self,
        *,
        file_id: str | None = None,
        file_path: str | None = None,
    ) -> CachedFile | None:
        clean_id = (file_id or "").strip()
        clean_path = _normalize_path(file_path)

        if clean_id:
            for cached in self.file_cache.values():
                if (cached.file_id or "").strip() == clean_id:
                    return cached

        if clean_path:
            if clean_path in self.file_cache:
                return self.file_cache[clean_path]
            for key, cached in self.file_cache.items():
                if _normalize_path(key) == clean_path or _normalize_path(cached.file_path) == clean_path:
                    return cached
            # Unique basename fallback (e.g. "intro.tex" -> "sections/intro.tex")
            basename = clean_path.rsplit("/", 1)[-1]
            matches = [
                cached
                for cached in self.file_cache.values()
                if _normalize_path(cached.file_path).rsplit("/", 1)[-1] == basename
            ]
            if len(matches) == 1:
                return matches[0]
        return None

    def put_cached_file(
        self,
        *,
        file_id: str | None,
        file_path: str,
        content: str,
    ) -> CachedFile:
        clean_path = _normalize_path(file_path) or "document.tex"
        existing = self.get_cached_file(file_id=file_id, file_path=clean_path)
        resolved_id = (file_id or "").strip() or (existing.file_id if existing else None)
        entry = CachedFile(file_id=resolved_id, file_path=clean_path, content=content)
        self.file_cache[clean_path] = entry
        return entry


class EditorToolResult(BaseModel):
    """Normalized execution output returned by every Editor tool."""

    model_config = ConfigDict(frozen=True)

    tool: str
    status: Literal["ok", "error", "rejected"] = "ok"
    data: Any = None
    error: str | None = None
    patch: EditorPatchProposal | None = None
    citations: tuple[CitationItem, ...] = ()


class EditorToolSpec(BaseModel):
    """Declarative specification for one intent-oriented Editor tool."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    args_schema: type[BaseModel]
    risk_tier: EditorRiskTier
    idempotent: bool
    handler: Any
    describe_confirmation: Callable[[dict[str, Any]], str] | None = None

    def format_confirmation(self, validated_args: dict[str, Any]) -> str:
        """Render a human-readable HITL confirmation prompt owned by this tool."""
        if self.describe_confirmation is not None:
            return self.describe_confirmation(validated_args)
        return f"Confirm executing page write action `{self.name}`"


class EditorToolRegistry(ToolRegistry[EditorToolSpec]):
    """OCP registry of editor tools used for planner formatting and policy gates."""

    label = "editor"
