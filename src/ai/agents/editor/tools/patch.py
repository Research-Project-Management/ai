"""Non-destructive anchor-verified patch proposal tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ai.agents.editor.patching.contracts import PatchKind
from ai.agents.editor.patching.pipeline import build_verified_patch
from ai.agents.editor.tools.contracts import (
    EditorToolResult,
    EditorToolRuntime,
    EditorToolSpec,
)
from ai.agents.editor.tools.file_ops import resolve_and_cache_file


class ProposeEditArgs(BaseModel):
    file_path: str | None = Field(
        default=None,
        description="Target file path (defaults to the active file in the editor).",
    )
    file_id: str | None = Field(default=None, description="Optional target file UUID.")
    kind: PatchKind = Field(
        default="replace",
        description="'replace' | 'insert_preamble' | 'insert_after' | 'insert_before'.",
    )
    search_text: str = Field(
        default="",
        description=(
            "Exact 2-5 verbatim lines to anchor against (without L### prefixes). "
            "Leave empty only when kind='insert_preamble'."
        ),
    )
    replace_text: str = Field(
        ...,
        description="Replacement or inserted LaTeX code.",
    )
    title: str = Field(..., min_length=1, description="Short human-readable change title.")
    rationale: str = Field(default="", description="Brief explanation of why this edit helps.")


async def handle_propose_edit(
    runtime: EditorToolRuntime,
    args: ProposeEditArgs,
) -> EditorToolResult:
    if len(runtime.patches) >= runtime.max_patches:
        return EditorToolResult(
            tool="propose_edit",
            status="error",
            error=f"Reached maximum patch proposals ({runtime.max_patches}) for one turn.",
        )

    target_path = (args.file_path or runtime.scope.active_file_path or "main.tex").lstrip("/")
    cached, err = await resolve_and_cache_file(
        runtime,
        file_id=args.file_id,
        file_path=target_path,
    )
    if err or cached is None:
        return EditorToolResult(
            tool="propose_edit",
            status="error",
            error=err or f"Could not resolve target file '{target_path}'.",
        )

    outcome = build_verified_patch(
        runtime.patches,
        content=cached.content,
        file_id=cached.file_id,
        file_path=cached.file_path,
        kind=args.kind,
        search_text=args.search_text,
        replace_text=args.replace_text,
        title=args.title,
        rationale=args.rationale,
        known_cite_keys=frozenset(runtime.known_cite_keys),
        known_labels=frozenset(runtime.known_labels),
    )
    if outcome.patch is None:
        return EditorToolResult(tool="propose_edit", status="error", error=outcome.error)

    final_patch = outcome.patch
    runtime.patches.append(final_patch)

    return EditorToolResult(
        tool="propose_edit",
        status="ok",
        patch=final_patch,
        data={
            "patch_id": final_patch.id,
            "file_path": final_patch.file_path,
            "from_line": final_patch.from_line,
            "to_line": final_patch.to_line,
            "match_strategy": final_patch.match_strategy,
            "warnings": list(final_patch.warnings),
        },
    )


def build_patch_tools() -> tuple[EditorToolSpec, ...]:
    """Build non-destructive anchor-verified patch tools."""
    return (
        EditorToolSpec(
            name="propose_edit",
            description=(
                "Propose an anchor-verified LaTeX diff preview for user review (does not overwrite files). "
                "Requires unique verbatim `search_text` unless `kind='insert_preamble'`."
            ),
            args_schema=ProposeEditArgs,
            risk_tier="propose",
            idempotent=True,
            handler=handle_propose_edit,
        ),
    )
