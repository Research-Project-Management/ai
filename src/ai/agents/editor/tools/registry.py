"""Default Editor tool registry builder."""

from __future__ import annotations

from ai.agents.editor.tools.analysis import build_analysis_tools
from ai.agents.editor.tools.contracts import EditorToolRegistry
from ai.agents.editor.tools.file_ops import build_file_tools
from ai.agents.editor.tools.patch import build_patch_tools
from ai.agents.editor.tools.table import build_table_tools
from ai.agents.editor.tools.write import build_write_tools
from ai.shared.context.facade import ResearchContextFacade
from ai.shared.github.contracts import GitHubArtifactGateway


def build_default_editor_tool_registry(
    *,
    artifact_client: GitHubArtifactGateway | None = None,
    context_facade: ResearchContextFacade | None = None,
) -> EditorToolRegistry:
    """Assemble File, Analysis, Propose-Edit, Table, and HITL-Write tools into an OCP registry."""
    return EditorToolRegistry(
        [
            *build_file_tools(),
            *build_analysis_tools(),
            *build_patch_tools(),
            *build_table_tools(
                artifact_client=artifact_client,
                context_facade=context_facade,
            ),
            *build_write_tools(),
        ]
    )
