"""Default Project tool registry assembling all 4 command groups via OCP."""

from __future__ import annotations

from ai.agents.project.tools.analytics_commands import build_analytics_tools
from ai.agents.project.tools.contracts import ProjectToolRegistry
from ai.agents.project.tools.destructive_commands import build_destructive_tools
from ai.agents.project.tools.read_commands import build_read_tools
from ai.agents.project.tools.write_commands import build_write_tools


def build_default_project_tool_registry() -> ProjectToolRegistry:
    """Assemble Read, Analytics, Write, and HITL-Destructive Project commands."""
    return ProjectToolRegistry(
        [
            *build_read_tools(),
            *build_analytics_tools(),
            *build_write_tools(),
            *build_destructive_tools(),
        ]
    )
