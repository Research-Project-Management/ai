"""Default Library tool registry builder assembling Read, Safe Write, and Destructive tools."""

from __future__ import annotations

from ai.agents.library.tools.contracts import LibraryToolRegistry
from ai.agents.library.tools.destructive import build_destructive_tools
from ai.agents.library.tools.read import build_read_tools
from ai.agents.library.tools.write import build_write_tools


def build_default_tool_registry() -> LibraryToolRegistry:
    """Assemble the default 11-tool registry for `library_agent`."""
    return LibraryToolRegistry(
        (
            *build_read_tools(),
            *build_write_tools(),
            *build_destructive_tools(),
        )
    )
