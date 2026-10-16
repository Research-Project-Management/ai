"""Viewport-based Context Projection strategies.

Applies the Strategy Pattern to dynamically project multi-domain context quotas
based on whether the user is in the LaTeX Editor, Kanban Board, Library, or Fullscreen Chat.
"""

from __future__ import annotations

from typing import Protocol

from ai.shared.context.contracts import EntityDomain, ViewportFocus


class ProjectionStrategy(Protocol):
    """Defines how context tokens are partitioned across research domains."""

    name: str

    def get_domain_weights(self) -> dict[EntityDomain, float]:
        """Return relative weights per domain summing to approximately 1.0."""
        ...


class EditorProjectionStrategy:
    """Strategy for page writing and editing: emphasizes Page AST and linked Experiments."""

    name = "editor"

    def get_domain_weights(self) -> dict[EntityDomain, float]:
        return {
            EntityDomain.PAGE: 0.55,
            EntityDomain.EXPERIMENT: 0.20,
            EntityDomain.LITERATURE: 0.15,
            EntityDomain.GOVERNANCE: 0.10,
        }


class KanbanProjectionStrategy:
    """Strategy for project and task tracking: emphasizes Governance and active experiment runs."""

    name = "kanban"

    def get_domain_weights(self) -> dict[EntityDomain, float]:
        return {
            EntityDomain.GOVERNANCE: 0.55,
            EntityDomain.EXPERIMENT: 0.20,
            EntityDomain.PAGE: 0.15,
            EntityDomain.LITERATURE: 0.10,
        }


class LibraryProjectionStrategy:
    """Strategy for literature exploration: emphasizes Papers, BibTeX, and linked claims."""

    name = "library"

    def get_domain_weights(self) -> dict[EntityDomain, float]:
        return {
            EntityDomain.LITERATURE: 0.55,
            EntityDomain.EXPERIMENT: 0.20,
            EntityDomain.PAGE: 0.15,
            EntityDomain.GOVERNANCE: 0.10,
        }


class GlobalChatProjectionStrategy:
    """Balanced strategy for fullscreen chat: macro project synthesis across all 4 domains."""

    name = "fullscreen_chat"

    def get_domain_weights(self) -> dict[EntityDomain, float]:
        return {
            EntityDomain.GOVERNANCE: 0.30,
            EntityDomain.LITERATURE: 0.30,
            EntityDomain.PAGE: 0.25,
            EntityDomain.EXPERIMENT: 0.15,
        }


class ContextProjector:
    """Registry and resolver for Viewport Projection Strategies."""

    def __init__(self) -> None:
        self._strategies: dict[str, ProjectionStrategy] = {
            "editor": EditorProjectionStrategy(),
            "kanban": KanbanProjectionStrategy(),
            "library": LibraryProjectionStrategy(),
            "fullscreen_chat": GlobalChatProjectionStrategy(),
        }
        self._default_strategy = GlobalChatProjectionStrategy()

    def resolve(self, focus: ViewportFocus) -> ProjectionStrategy:
        """Resolve the appropriate projection strategy from user viewport."""
        return self._strategies.get(focus.view, self._default_strategy)
