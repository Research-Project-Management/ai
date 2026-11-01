"""Backend payload -> Library DTO mapping helpers."""

from __future__ import annotations

from typing import Any

from ai.agents.library.gateway.contracts import CollectionNode, ItemSummary


def _extract_authors(raw: dict[str, Any]) -> tuple[str, ...]:
    names: list[str] = []
    for key in ("authors", "contributors", "creators"):
        entries = raw.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, str) and entry.strip():
                names.append(entry.strip())
            elif isinstance(entry, dict):
                full = str(
                    entry.get("fullName")
                    or entry.get("name")
                    or f"{entry.get('firstName', '')} {entry.get('lastName', '')}"
                ).strip()
                if full:
                    names.append(full)
        if names:
            return tuple(names[:3])

    single = str(raw.get("author") or "").strip()
    if single:
        return tuple(part.strip() for part in single.split(",")[:3] if part.strip())
    return ()


def _extract_label_list(raw_entries: Any, *, nested_key: str) -> tuple[str, ...]:
    if not isinstance(raw_entries, list):
        return ()
    labels: list[str] = []
    for entry in raw_entries:
        if isinstance(entry, str) and entry.strip():
            labels.append(entry.strip())
        elif isinstance(entry, dict):
            nested = entry.get(nested_key)
            if isinstance(nested, dict) and str(nested.get("name") or "").strip():
                labels.append(str(nested["name"]).strip())
            elif str(entry.get("name") or entry.get("title") or "").strip():
                labels.append(str(entry.get("name") or entry.get("title")).strip())
    return tuple(dict.fromkeys(labels))


def project_summary(raw: dict[str, Any]) -> ItemSummary:
    venue = str(
        raw.get("publicationTitle")
        or raw.get("journal")
        or raw.get("proceedingsTitle")
        or raw.get("conferenceName")
        or raw.get("publisher")
        or ""
    ).strip()
    year_val = raw.get("year")
    parsed_year = int(year_val) if isinstance(year_val, int) else None
    if parsed_year is None and isinstance(year_val, str) and year_val.strip().isdigit():
        parsed_year = int(year_val.strip())

    return ItemSummary(
        id=str(raw.get("id") or ""),
        title=str(raw.get("title") or "Untitled").strip() or "Untitled",
        authors=_extract_authors(raw),
        year=parsed_year,
        venue=venue,
        item_type=str(raw.get("itemType") or raw.get("type") or ""),
        doi=str(raw["doi"]).strip() if raw.get("doi") else None,
        arxiv_id=str(raw["arxivId"]).strip() if raw.get("arxivId") else None,
        url=str(raw["url"]).strip() if raw.get("url") else None,
        read_status=str(raw.get("readStatus") or ""),
        tags=_extract_label_list(raw.get("tags"), nested_key="tag"),
        collections=_extract_label_list(
            raw.get("collectionItems") or raw.get("collections"),
            nested_key="collection",
        ),
    )


def project_collections(raw_nodes: Any) -> tuple[CollectionNode, ...]:
    if not isinstance(raw_nodes, list):
        return ()
    nodes: list[CollectionNode] = []
    for entry in raw_nodes[:50]:
        if not isinstance(entry, dict):
            continue
        raw_count = entry.get("itemCount")
        if raw_count is None and isinstance(entry.get("_count"), dict):
            raw_count = entry["_count"].get("items", 0)
        nodes.append(
            CollectionNode(
                id=str(entry.get("id") or ""),
                name=str(entry.get("name") or "").strip(),
                parent_id=str(entry["parentId"]) if entry.get("parentId") else None,
                item_count=int(raw_count or 0),
                children=project_collections(entry.get("children")),
            )
        )
    return tuple(nodes)


def normalize_read_status(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().lower()
    if cleaned == "read":
        return "completed"
    return cleaned or None
