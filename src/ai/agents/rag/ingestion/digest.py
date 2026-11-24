"""Episodic Timeline Digest builder and upserter (CoALA Episodic Tier / ADR-001 Phase 3)."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from ai.shared.contracts.knowledge import KnowledgeDocumentInput, KnowledgeStore


class TimelineEventInput(BaseModel):
    """Canonical project event summarized inside a periodic timeline digest."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(..., min_length=1)
    source_type: str = Field(default="project_doc", min_length=1)
    title: str = Field(..., min_length=1)
    effective_at: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    lifecycle_status: str = "active"
    supersedes_source_id: str | None = None


class TimelineDigestResult(BaseModel):
    """Result summary emitted after indexing a periodic timeline digest."""

    model_config = ConfigDict(frozen=True)

    space_id: str
    digest_source_id: str
    period_label: str
    events_included: int
    chunks_indexed: int
    markdown: str


def build_timeline_digest_markdown(
    period_label: str,
    events: Sequence[TimelineEventInput],
) -> str:
    """Render events chronologically into a compact CoALA episodic digest."""
    ordered = sorted(events, key=lambda item: (item.effective_at.strip(), item.source_id.strip()))
    lines: list[str] = [f"# Project Timeline Digest: {period_label.strip()}"]
    for event in ordered:
        line = (
            f"- [{event.effective_at.strip()}] ({event.source_type.strip()}, "
            f"{event.lifecycle_status.strip().lower()}) **{event.title.strip()}** "
            f"[`{event.source_id.strip()}`]: {event.summary.strip()}"
        )
        if event.supersedes_source_id and event.supersedes_source_id.strip():
            line += f" (Supersedes: `{event.supersedes_source_id.strip()}`)"
        lines.append(line)
    return "\n".join(lines) + "\n"


async def upsert_timeline_digest(
    store: KnowledgeStore,
    *,
    space_id: str,
    digest_source_id: str,
    period_label: str,
    events: Sequence[TimelineEventInput],
) -> TimelineDigestResult:
    """Validate events, compile chronological Markdown, and index as `timeline_digest`."""
    clean_space = space_id.strip()
    clean_digest_id = digest_source_id.strip()
    clean_period = period_label.strip()
    if not clean_space or not clean_digest_id or not clean_period:
        raise ValueError("space_id, digest_source_id, and period_label must be non-empty")
    if not events:
        raise ValueError("At least one TimelineEventInput is required to build a timeline digest")

    ordered = sorted(events, key=lambda item: (item.effective_at.strip(), item.source_id.strip()))
    latest_effective_at = ordered[-1].effective_at.strip()
    markdown = build_timeline_digest_markdown(clean_period, ordered)
    chunks_indexed = await store.upsert(
        KnowledgeDocumentInput(
            space_id=clean_space,
            source_type="timeline_digest",
            source_id=clean_digest_id,
            title=f"Project Timeline Digest: {clean_period}",
            content=markdown,
            metadata={
                "valid_from": latest_effective_at,
                "effective_at": latest_effective_at,
                "lifecycle_status": "active",
                "period_label": clean_period,
                "events_count": len(ordered),
            },
        )
    )
    return TimelineDigestResult(
        space_id=clean_space,
        digest_source_id=clean_digest_id,
        period_label=clean_period,
        events_included=len(ordered),
        chunks_indexed=chunks_indexed,
        markdown=markdown,
    )
