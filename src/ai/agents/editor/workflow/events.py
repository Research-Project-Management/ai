"""Editor tool events: shared tool telemetry plus live patch previews."""

from __future__ import annotations

from typing import Protocol

from langchain_core.runnables import RunnableConfig

from ai.agents.editor.patching.contracts import EditorPatchProposal
from ai.shared.workflow.events import (
    PATCH_PREVIEW,
    LangGraphToolEvents,
    NullToolEvents,
    ToolEventPublisher,
)


class EditorEventPublisher(ToolEventPublisher, Protocol):
    """Port for tool telemetry and patch preview notifications."""

    async def emit_patch_preview(
        self,
        patch: EditorPatchProposal,
        total_patches: int,
        config: RunnableConfig,
    ) -> None: ...


class LangGraphEditorEvents(LangGraphToolEvents):
    """Tool events plus `editor.patch_preview` for downstream SSE translation."""

    def __init__(self) -> None:
        super().__init__("editor_agent")

    async def emit_patch_preview(
        self,
        patch: EditorPatchProposal,
        total_patches: int,
        config: RunnableConfig,
    ) -> None:
        await self.dispatch(
            PATCH_PREVIEW,
            {"patch": patch.model_dump(mode="json"), "total_patches": total_patches},
            config,
        )


class NullEditorEvents(NullToolEvents):
    """No-op publisher for isolated non-streaming unit tests."""

    async def emit_patch_preview(
        self,
        patch: EditorPatchProposal,
        total_patches: int,
        config: RunnableConfig,
    ) -> None:
        del patch, total_patches, config
