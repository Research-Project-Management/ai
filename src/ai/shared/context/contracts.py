"""Domain contracts, DTOs, and protocols for the Research Context & Provenance Engine.

Grounded in the W3C PROV-DM standard and multi-domain academic research workflows.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class EntityDomain(StrEnum):
    """The 4 core domains of the unified scientific research ecosystem."""

    GOVERNANCE = "gov"  # Tasks, Milestones, Members
    LITERATURE = "lit"  # Papers, BibTeX, Notes, Collections
    EXPERIMENT = "exp"  # Datasets, Runs, Hyperparameters, Metrics
    PAGE = "page"  # Pages, Sections, Tables, Figures, Equations


class RelationType(StrEnum):
    """W3C PROV-inspired semantic relations linking heterogeneous scientific entities."""

    INFORMS = "informs"  # Paper -> Task (Literature informs research direction)
    TRACKS = "tracks"  # Task -> Experiment Run (Task tracks experiment run)
    PROVES = "proves"  # Experiment Run -> Page (Run generates table/metric in Page)
    CITES = "cites"  # Page -> Paper (Page cites bibliographic reference)
    DEPENDS_ON = "depends_on"  # Entity depends on another entity within/across domains
    DERIVED_FROM = "derived_from"  # Metric/Figure derived from Dataset/Run


class GraphNode(BaseModel):
    """A lightweight attributed vertex in the scientific provenance graph."""

    id: str
    domain: EntityDomain
    label: str
    attributes: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class GraphEdge(BaseModel):
    """A directed semantic edge connecting two provenance entities."""

    src_id: str
    dst_id: str
    relation: RelationType
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class ViewportFocus(BaseModel):
    """Represents the researcher's active operational viewport across Flux UI."""

    view: Literal["editor", "kanban", "library", "fullscreen_chat"] = "fullscreen_chat"
    target_id: str | None = None
    selection: str | None = None
    attached_sources: tuple[str, ...] = Field(default_factory=tuple)
    inferred_entity_id: str | None = None

    model_config = {"frozen": True}


class ContextEnvelope(BaseModel):
    """Boundary-anchored prompt envelope structured to eliminate the 'Lost in the Middle' effect.

    Follows Stanford TACL 2024:
      - Primacy: High-attention prefix (Research Identity, Venue, Hard Constraints).
      - Middle: Supplementary entity network & retrieved evidence chunks.
      - Recency: High-attention suffix (Active Viewport Selection, Working Query).
    """

    primacy_text: str
    middle_text: str
    recency_text: str
    total_tokens: int
    included_nodes: tuple[GraphNode, ...] = Field(default_factory=tuple)

    @property
    def full_prompt(self) -> str:
        """Compose the full structured prompt with clean boundary delimiters."""
        sections = [
            self.primacy_text.strip(),
            self.middle_text.strip(),
            self.recency_text.strip(),
        ]
        return "\n\n".join(s for s in sections if s)


class AuditFinding(BaseModel):
    """Represents a cross-domain scientific inconsistency, gap, or drift finding."""

    rule_id: str
    severity: Literal["warning", "error", "info"]
    message: str
    entity_ids: tuple[str, ...] = Field(default_factory=tuple)

    model_config = {"frozen": True}


def extract_viewport_focus(context: Mapping[str, Any] | None) -> ViewportFocus:
    """Extract a typed ViewportFocus from inbound raw request context."""
    if not context:
        return ViewportFocus(view="fullscreen_chat")

    raw_view = str(context.get("viewport") or "").strip().lower()
    page_id = str(context.get("page_id") or "").strip()
    file_path = str(context.get("active_file_path") or context.get("filename") or "").strip()
    project_id = str(context.get("project_id") or "").strip()
    selection = str(context.get("selection") or "").strip() or None

    attached = context.get("attached_sources")
    attached_ids: list[str] = []
    if isinstance(attached, list):
        for item in attached:
            if isinstance(item, dict):
                sid = item.get("source_id") or item.get("item_id") or item.get("id")
                if sid:
                    attached_ids.append(str(sid))
            elif isinstance(item, str) and item.strip():
                attached_ids.append(item.strip())

    doc_ids = context.get("document_ids")
    if isinstance(doc_ids, list):
        for d in doc_ids:
            if isinstance(d, str) and d.strip() and d.strip() not in attached_ids:
                attached_ids.append(d.strip())

    if raw_view in ("editor", "kanban", "library", "fullscreen_chat"):
        view = raw_view
    elif file_path or page_id:
        view = "editor"
    elif project_id and not page_id:
        view = "kanban"
    elif attached_ids and not file_path:
        view = "library"
    else:
        view = "fullscreen_chat"

    target_id = file_path or page_id or project_id or (attached_ids[0] if attached_ids else None)

    return ViewportFocus(
        view=view,  # type: ignore[arg-type]
        target_id=target_id,
        selection=selection,
        attached_sources=tuple(attached_ids),
        inferred_entity_id=str(context.get("inferred_entity_id") or "").strip() or None,
    )
