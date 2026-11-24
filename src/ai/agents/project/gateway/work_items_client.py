"""Sub-client handling work items CRUD, workflow transitions, assignments & comments."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Literal

from ai.agents.project.gateway.contracts import (
    ProjectGatewayError,
    ProjectScope,
    WorkItemComment,
    WorkItemSummary,
)
from ai.agents.project.gateway.transport import (
    as_dict,
    encode_seg,
    unwrap_nested_dict,
)
from ai.shared.gateway.transport import BackendTransport


def _compute_is_overdue(due_date: str | None, completed: bool, explicit: Any) -> bool:
    if isinstance(explicit, bool):
        return explicit and not completed
    if completed or not due_date:
        return False
    raw = due_date.strip()
    try:
        parsed = datetime.fromisoformat(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed < datetime.now(UTC)
    except ValueError:
        return False


def _parse_work_item(item: dict[str, Any]) -> WorkItemSummary:
    state_obj = as_dict(item.get("state"))
    assignee_obj = as_dict(item.get("assignee"))
    project_obj = as_dict(item.get("project"))

    raw_assignees = item.get("assignees")
    assignee_ids_list: list[str] = []
    assignee_names_list: list[str] = []
    if isinstance(raw_assignees, list):
        for entry in raw_assignees:
            if isinstance(entry, dict):
                uid = str(entry.get("id") or entry.get("userId") or "").strip()
                uname = str(entry.get("name") or entry.get("email") or "").strip()
                if uid:
                    assignee_ids_list.append(uid)
                if uname:
                    assignee_names_list.append(uname)

    assignee_id_raw = item.get("assigneeId") or assignee_obj.get("id")
    primary_assignee_id = str(assignee_id_raw) if assignee_id_raw else None
    if primary_assignee_id and primary_assignee_id not in assignee_ids_list:
        assignee_ids_list.insert(0, primary_assignee_id)

    primary_assignee_name = str(
        assignee_obj.get("name")
        or item.get("assigneeName")
        or (assignee_names_list[0] if assignee_names_list else "")
    ).strip()

    raw_labels = item.get("labels")
    label_names: list[str] = []
    label_ids: list[str] = []
    if isinstance(raw_labels, list):
        for lb in raw_labels:
            if isinstance(lb, dict):
                lname = str(lb.get("name") or "").strip()
                lid = str(lb.get("id") or "").strip()
                if lname:
                    label_names.append(lname)
                if lid:
                    label_ids.append(lid)
            elif isinstance(lb, str) and lb.strip():
                label_names.append(lb.strip())

    completed = bool(item.get("completed"))
    due_date = str(item["dueDate"]) if item.get("dueDate") else None
    col_id = (
        str(item["columnId"])
        if item.get("columnId")
        else (str(item["stateId"]) if item.get("stateId") else None)
    )

    return WorkItemSummary(
        id=str(item.get("id") or ""),
        identifier=str(item.get("identifier") or "").strip(),
        title=str(item.get("title") or "").strip(),
        description=str(item.get("description") or item.get("content") or "").strip()[:1000],
        project_id=str(item.get("projectId") or project_obj.get("id") or ""),
        project_name=str(project_obj.get("name") or "").strip(),
        column_id=col_id,
        state_name=str(state_obj.get("name") or item.get("stateName") or "").strip(),
        state_group=str(item.get("stateGroup") or state_obj.get("group") or "").strip().lower(),
        priority=str(item.get("priority") or "none").lower(),
        completed=completed,
        is_archived=bool(item.get("archivedAt")),
        assignee_id=primary_assignee_id,
        assignee_name=primary_assignee_name,
        assignee_ids=tuple(assignee_ids_list),
        assignee_names=tuple(assignee_names_list),
        labels=tuple(label_names),
        label_ids=tuple(label_ids),
        start_date=str(item["startDate"]) if item.get("startDate") else None,
        due_date=due_date,
        is_overdue=_compute_is_overdue(due_date, completed, item.get("isOverdue")),
        parent_work_item_id=str(item["parentWorkItemId"]) if item.get("parentWorkItemId") else None,
        child_count=int(item.get("childWorkItemCount") or 0),
        created_at=str(item["createdAt"]) if item.get("createdAt") else None,
        updated_at=str(item["updatedAt"]) if item.get("updatedAt") else None,
    )


def _extract_work_item_dict(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ProjectGatewayError(502, "Unexpected work item payload format from Backend")
    for key in ("workItem", "item"):
        nested = raw.get(key)
        if isinstance(nested, dict):
            return nested
    return raw


def _parse_comment(item: dict[str, Any]) -> WorkItemComment:
    author = as_dict(item.get("author"))
    return WorkItemComment(
        id=str(item.get("id") or ""),
        work_item_id=str(item.get("workItemId") or ""),
        author_id=str(item.get("authorId") or item.get("createdById") or author.get("id") or ""),
        author_name=str(author.get("name") or author.get("email") or "").strip(),
        content=str(item.get("content") or "").strip(),
        created_at=str(item["createdAt"]) if item.get("createdAt") else None,
    )


class WorkItemsClient:
    """Adapter for `/api/v1/work-items/...`, `/api/v1/projects/:id/work-items/...` & `/your-work`."""

    def __init__(self, transport: BackendTransport) -> None:
        self._transport = transport

    async def list_work_items(
        self,
        scope: ProjectScope,
        project_id: str,
        *,
        column_id: str | None = None,
        state_group: str | None = None,
        priority: str | None = None,
        assignee_id: str | None = None,
        label: str | None = None,
        completed: bool | None = None,
        search: str = "",
        limit: int = 50,
    ) -> list[WorkItemSummary]:
        del scope
        params: dict[str, Any] = {
            "columnId": column_id.strip() if column_id else None,
            "stateGroup": state_group.strip().lower() if state_group else None,
            "priority": priority.strip().lower() if priority else None,
            "assigneeId": assignee_id.strip() if assignee_id else None,
            "labels": label.strip() if label else None,
            "search": search.strip() or None,
            "limit": max(1, min(limit, 200)),
        }
        if completed is not None:
            params["completed"] = "true" if completed else "false"

        raw = await self._transport.request(
            "GET",
            f"/api/v1/projects/{encode_seg(project_id)}/work-items",
            params=params,
        )
        entries = raw.get("workItems") or raw.get("items") or [] if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            return []
        return [
            _parse_work_item(item) for item in entries if isinstance(item, dict) and item.get("id")
        ]

    async def get_my_work_items(
        self,
        scope: ProjectScope,
        *,
        project_id: str | None = None,
        relation: Literal["assigned", "created", "subscribed", "all"] = "assigned",
    ) -> list[WorkItemSummary]:
        del scope
        path = (
            f"/api/v1/analytics/projects/{encode_seg(project_id)}/your-work"
            if project_id and project_id.strip()
            else "/api/v1/analytics/your-work"
        )
        raw = await self._transport.request("GET", path)
        if not isinstance(raw, dict):
            return []

        buckets = ("assigned", "created", "subscribed") if relation == "all" else (relation,)
        seen_ids: set[str] = set()
        items: list[WorkItemSummary] = []
        for bucket in buckets:
            raw_list = raw.get(bucket)
            if not isinstance(raw_list, list):
                continue
            for entry in raw_list:
                if not isinstance(entry, dict) or not entry.get("id"):
                    continue
                parsed = _parse_work_item(entry)
                if parsed.id not in seen_ids:
                    seen_ids.add(parsed.id)
                    items.append(parsed)
        return items

    async def get_work_item(self, scope: ProjectScope, work_item_id: str) -> WorkItemSummary:
        del scope
        raw = await self._transport.request(
            "GET",
            f"/api/v1/work-items/{encode_seg(work_item_id)}",
        )
        return _parse_work_item(_extract_work_item_dict(raw))

    async def create_work_item(
        self,
        scope: ProjectScope,
        project_id: str,
        *,
        title: str,
        description: str | None = None,
        column_id: str | None = None,
        assignee_id: str | None = None,
        priority: str | None = None,
        due_date: str | None = None,
        start_date: str | None = None,
        label_ids: Sequence[str] = (),
        labels: Sequence[str] = (),
        parent_work_item_id: str | None = None,
    ) -> WorkItemSummary:
        del scope
        body: dict[str, Any] = {"title": title.strip()}
        if description is not None:
            body["description"] = description
        if column_id:
            body["columnId"] = column_id.strip()
        if assignee_id:
            body["assigneeId"] = assignee_id.strip()
        if priority:
            body["priority"] = priority.strip().lower()
        if due_date:
            body["dueDate"] = due_date.strip()
        if start_date:
            body["startDate"] = start_date.strip()
        if label_ids:
            body["labelIds"] = [lid.strip() for lid in label_ids if lid.strip()]
        if labels:
            body["labels"] = [lbl.strip() for lbl in labels if lbl.strip()]
        if parent_work_item_id:
            body["parentWorkItemId"] = parent_work_item_id.strip()

        raw = await self._transport.request(
            "POST",
            f"/api/v1/projects/{encode_seg(project_id)}/work-items",
            json_body=body,
        )
        return _parse_work_item(_extract_work_item_dict(raw))

    async def update_work_item(
        self,
        scope: ProjectScope,
        work_item_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        column_id: str | None = None,
        assignee_id: str | None = None,
        clear_assignee: bool = False,
        priority: str | None = None,
        due_date: str | None = None,
        start_date: str | None = None,
        completed: bool | None = None,
        label_ids: Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
    ) -> WorkItemSummary:
        del scope
        body: dict[str, Any] = {}
        if title is not None:
            body["title"] = title.strip()
        if description is not None:
            body["description"] = description
        if column_id is not None:
            body["columnId"] = column_id.strip()
        if clear_assignee:
            body["assigneeId"] = None
        elif assignee_id is not None:
            body["assigneeId"] = assignee_id.strip()
        if priority is not None:
            body["priority"] = priority.strip().lower()
        if due_date is not None:
            body["dueDate"] = due_date
        if start_date is not None:
            body["startDate"] = start_date
        if completed is not None:
            body["completed"] = completed
        if label_ids is not None:
            body["labelIds"] = list(label_ids)
        if labels is not None:
            body["labels"] = list(labels)

        raw = await self._transport.request(
            "PUT",
            f"/api/v1/work-items/{encode_seg(work_item_id)}",
            json_body=body,
        )
        return _parse_work_item(_extract_work_item_dict(raw))

    async def delete_work_item(self, scope: ProjectScope, work_item_id: str) -> dict[str, Any]:
        del scope
        raw = await self._transport.request(
            "DELETE",
            f"/api/v1/work-items/{encode_seg(work_item_id)}",
        )
        return raw if isinstance(raw, dict) else {"message": "Work item deleted"}

    async def archive_work_item(self, scope: ProjectScope, work_item_id: str) -> WorkItemSummary:
        del scope
        raw = await self._transport.request(
            "POST",
            f"/api/v1/work-items/{encode_seg(work_item_id)}/archive",
        )
        return _parse_work_item(_extract_work_item_dict(raw))

    async def assign_work_item(
        self,
        scope: ProjectScope,
        project_id: str,
        work_item_id: str,
        *,
        assignee_id: str | None = None,
    ) -> dict[str, Any]:
        del scope
        base = f"/api/v1/projects/{encode_seg(project_id)}/work-items/{encode_seg(work_item_id)}"
        if assignee_id:
            raw = await self._transport.request(
                "POST",
                f"{base}/assign",
                json_body={"assigneeId": assignee_id.strip()},
            )
        else:
            raw = await self._transport.request("POST", f"{base}/unassign")
        return raw if isinstance(raw, dict) else {"newAssigneeId": assignee_id}

    async def list_comments(self, scope: ProjectScope, work_item_id: str) -> list[WorkItemComment]:
        del scope
        raw = await self._transport.request(
            "GET",
            f"/api/v1/work-items/{encode_seg(work_item_id)}/comments",
        )
        entries = raw.get("comments", []) if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            return []
        return [_parse_comment(c) for c in entries if isinstance(c, dict)]

    async def add_comment(
        self,
        scope: ProjectScope,
        work_item_id: str,
        *,
        content: str,
        project_id: str | None = None,
    ) -> WorkItemComment:
        del scope
        body: dict[str, Any] = {"content": content.strip()}
        if project_id:
            body["projectId"] = project_id.strip()
        raw = await self._transport.request(
            "POST",
            f"/api/v1/work-items/{encode_seg(work_item_id)}/comments",
            json_body=body,
        )
        return _parse_comment(unwrap_nested_dict(raw, "comment", {"content": content}))
