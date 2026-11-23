"""Sub-client handling project metadata, members, workflow columns, labels & activity feeds."""

from __future__ import annotations

from typing import Any, Literal

from ai.agents.project.gateway.contracts import (
    ActivityEntry,
    ProjectGatewayError,
    ProjectLabel,
    ProjectMember,
    ProjectOverview,
    ProjectScope,
    ProjectSummary,
    WorkflowState,
)
from ai.agents.project.gateway.transport import (
    ProjectHttpTransport,
    as_dict,
    encode_seg,
    unwrap_nested_dict,
)


def _parse_project(item: dict[str, Any]) -> ProjectSummary:
    state_obj = item.get("state")
    state_name = (
        str(state_obj.get("name") or "")
        if isinstance(state_obj, dict)
        else str(item.get("stateName") or "")
    ).strip()
    members = item.get("members")
    count_obj = item.get("_count")
    member_count = (
        int(count_obj["members"])
        if isinstance(count_obj, dict) and isinstance(count_obj.get("members"), int)
        else len(members)
        if isinstance(members, list)
        else int(item.get("totalMembers") or 0)
    )
    return ProjectSummary(
        id=str(item.get("id") or ""),
        name=str(item.get("name") or "").strip(),
        identifier=str(item.get("identifier") or "").strip(),
        description=str(item.get("description") or "").strip(),
        priority=str(item.get("priority") or "none").lower(),
        state_name=state_name,
        state_id=str(item["stateId"]) if item.get("stateId") else None,
        start_date=str(item["startDate"]) if item.get("startDate") else None,
        target_date=str(item["targetDate"]) if item.get("targetDate") else None,
        is_archived=bool(item.get("isArchived")),
        your_role=str(item.get("yourRole") or "").strip(),
        member_count=member_count,
    )


def _parse_member(item: dict[str, Any]) -> ProjectMember:
    user_obj = as_dict(item.get("user"))
    uid = str(item.get("userId") or user_obj.get("id") or item.get("id") or "").strip()
    name = str(user_obj.get("name") or item.get("name") or "").strip()
    email = str(user_obj.get("email") or item.get("email") or "").strip()
    avatar = user_obj.get("avatar") or item.get("avatar")
    return ProjectMember(
        user_id=uid,
        name=name or email or uid,
        email=email,
        avatar=str(avatar) if avatar else None,
        role=str(item.get("role") or "contributor").lower(),
    )


def _parse_workflow_state(item: dict[str, Any]) -> WorkflowState:
    return WorkflowState(
        id=str(item.get("id") or ""),
        name=str(item.get("name") or item.get("title") or "").strip(),
        group=str(item.get("group") or "unstarted").lower(),
        color=str(item.get("color") or item.get("accentColor") or ""),
        sequence=int(item["sequence"]) if isinstance(item.get("sequence"), int) else 0,
        is_default=bool(item.get("isDefault")),
    )


def _parse_label(item: dict[str, Any]) -> ProjectLabel:
    return ProjectLabel(
        id=str(item.get("id") or ""),
        name=str(item.get("name") or "").strip(),
        color=str(item.get("color") or ""),
        description=str(item.get("description") or "").strip(),
        type=str(item.get("type") or "work_item"),
        parent_id=str(item["parentId"]) if item.get("parentId") else None,
    )


def _parse_activity(item: dict[str, Any]) -> ActivityEntry:
    actor = as_dict(item.get("actor"))
    project = as_dict(item.get("project"))
    return ActivityEntry(
        id=str(item.get("id") or ""),
        entity_type=str(item.get("entityType") or ""),
        entity_id=str(item.get("entityId") or ""),
        verb=str(item.get("verb") or ""),
        field=str(item["field"]) if item.get("field") is not None else None,
        old_value=str(item["oldValue"]) if item.get("oldValue") is not None else None,
        new_value=str(item["newValue"]) if item.get("newValue") is not None else None,
        actor_name=str(actor.get("name") or actor.get("email") or "").strip(),
        project_name=str(project.get("name") or "").strip(),
        created_at=str(item["createdAt"]) if item.get("createdAt") else None,
    )


class ProjectsClient:
    """Adapter for `/api/v1/projects/...` and `/api/.../activity/feed`."""

    def __init__(self, transport: ProjectHttpTransport) -> None:
        self._transport = transport

    async def list_projects(
        self,
        scope: ProjectScope,
        *,
        project_type: Literal["created", "shared", "all"] = "all",
        search: str = "",
        priority: str | None = None,
        is_archived: bool | None = None,
    ) -> list[ProjectSummary]:
        del scope
        params: dict[str, Any] = {
            "type": project_type,
            "search": search.strip() or None,
            "priority": priority.strip().lower() if priority else None,
        }
        if is_archived is not None:
            params["isArchived"] = "true" if is_archived else "false"

        raw = await self._transport.request("GET", "/api/v1/projects", params=params)
        entries = raw.get("projects", []) if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            return []
        return [
            _parse_project(item) for item in entries if isinstance(item, dict) and item.get("id")
        ]

    async def get_project(self, scope: ProjectScope, project_id: str) -> ProjectSummary:
        del scope
        raw = await self._transport.request("GET", f"/api/v1/projects/{encode_seg(project_id)}")
        if not isinstance(raw, dict):
            raise ProjectGatewayError(502, "Unexpected project detail payload from Backend")
        proj_obj = unwrap_nested_dict(raw, "project")
        parsed = _parse_project(proj_obj)
        if not parsed.your_role and raw.get("yourRole"):
            return parsed.model_copy(update={"your_role": str(raw["yourRole"])})
        return parsed

    async def get_project_overview(self, scope: ProjectScope, project_id: str) -> ProjectOverview:
        del scope
        raw = await self._transport.request(
            "GET",
            f"/api/v1/projects/{encode_seg(project_id)}/overview",
        )
        if not isinstance(raw, dict):
            raise ProjectGatewayError(502, "Unexpected project overview payload from Backend")

        proj = as_dict(raw.get("project"))
        metrics = as_dict(raw.get("metrics"))
        assets = as_dict(raw.get("assets"))
        lead = as_dict(proj.get("lead"))
        state = as_dict(proj.get("state"))
        upd = as_dict(raw.get("currentUpdate"))
        recent_raw = raw.get("recentActivities")
        recent = (
            [r for r in recent_raw if isinstance(r, dict)][:10]
            if isinstance(recent_raw, list)
            else []
        )

        return ProjectOverview(
            project_id=str(proj.get("id") or project_id),
            name=str(proj.get("name") or "").strip(),
            identifier=str(proj.get("identifier") or "").strip(),
            description=str(proj.get("description") or "").strip(),
            priority=str(proj.get("priority") or "none").lower(),
            state_name=str(state.get("name") or "").strip(),
            total_members=int(proj.get("totalMembers") or 0),
            lead_name=str(lead.get("name") or "").strip(),
            total_work_items=int(metrics.get("totalWorkItems") or metrics.get("totalIssues") or 0),
            completed=int(metrics.get("completed") or 0),
            started=int(metrics.get("started") or 0),
            unstarted=int(metrics.get("unstarted") or 0),
            backlog=int(metrics.get("backlog") or 0),
            cancelled=int(metrics.get("cancelled") or 0),
            overdue=int(metrics.get("overdue") or 0),
            completion_percentage=float(metrics.get("completionPercentage") or 0.0),
            pages_count=int(assets.get("pagesCount") or 0),
            library_count=int(assets.get("libraryCount") or 0),
            current_update_status=str(upd["status"]) if upd.get("status") else None,
            current_update_message=str(upd["message"]) if upd.get("message") else None,
            recent_activities=recent,
        )

    async def update_project(
        self,
        scope: ProjectScope,
        project_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        priority: str | None = None,
        start_date: str | None = None,
        target_date: str | None = None,
        is_archived: bool | None = None,
    ) -> ProjectSummary:
        del scope
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name.strip()
        if description is not None:
            body["description"] = description
        if priority is not None:
            body["priority"] = priority.strip().lower()
        if start_date is not None:
            body["startDate"] = start_date
        if target_date is not None:
            body["targetDate"] = target_date
        if is_archived is not None:
            body["isArchived"] = is_archived

        raw = await self._transport.request(
            "PUT",
            f"/api/v1/projects/{encode_seg(project_id)}",
            json_body=body,
        )
        return _parse_project(unwrap_nested_dict(raw, "project", {"id": project_id}))

    async def list_members(
        self,
        scope: ProjectScope,
        project_id: str,
        *,
        role: str | None = None,
        search: str = "",
        limit: int = 50,
    ) -> list[ProjectMember]:
        del scope
        raw = await self._transport.request(
            "GET",
            f"/api/v1/projects/{encode_seg(project_id)}/members",
            params={
                "role": role.strip().lower() if role else None,
                "search": search.strip() or None,
                "limit": max(1, min(limit, 100)),
            },
        )
        entries = raw.get("members", []) if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            return []
        return [_parse_member(m) for m in entries if isinstance(m, dict)]

    async def add_member(
        self,
        scope: ProjectScope,
        project_id: str,
        *,
        user_id: str,
        role: str = "contributor",
    ) -> ProjectMember:
        del scope
        raw = await self._transport.request(
            "POST",
            f"/api/v1/projects/{encode_seg(project_id)}/members",
            json_body={"userId": user_id.strip(), "role": role.strip().lower()},
        )
        return _parse_member(unwrap_nested_dict(raw, "member", {"userId": user_id, "role": role}))

    async def update_member_role(
        self,
        scope: ProjectScope,
        project_id: str,
        user_id: str,
        *,
        role: str,
    ) -> ProjectMember:
        del scope
        raw = await self._transport.request(
            "PUT",
            f"/api/v1/projects/{encode_seg(project_id)}/members/{encode_seg(user_id)}",
            json_body={"role": role.strip().lower()},
        )
        return _parse_member(unwrap_nested_dict(raw, "member", {"userId": user_id, "role": role}))

    async def remove_member(
        self, scope: ProjectScope, project_id: str, user_id: str
    ) -> dict[str, Any]:
        del scope
        raw = await self._transport.request(
            "DELETE",
            f"/api/v1/projects/{encode_seg(project_id)}/members/{encode_seg(user_id)}",
        )
        return raw if isinstance(raw, dict) else {"message": "Member removed"}

    async def list_workflow_states(
        self, scope: ProjectScope, project_id: str
    ) -> list[WorkflowState]:
        del scope
        raw = await self._transport.request(
            "GET",
            f"/api/v1/projects/{encode_seg(project_id)}/states",
        )
        entries = raw.get("states") or raw.get("columns") or [] if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            return []
        states = [_parse_workflow_state(s) for s in entries if isinstance(s, dict) and s.get("id")]
        return sorted(states, key=lambda s: s.sequence)

    async def list_labels(self, scope: ProjectScope, project_id: str) -> list[ProjectLabel]:
        del scope
        raw = await self._transport.request(
            "GET",
            f"/api/v1/projects/{encode_seg(project_id)}/labels",
            params={"type": "work_item"},
        )
        entries = raw.get("labels", []) if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            return []
        return [_parse_label(lb) for lb in entries if isinstance(lb, dict) and lb.get("id")]

    async def create_label(
        self,
        scope: ProjectScope,
        project_id: str,
        *,
        name: str,
        color: str | None = None,
        description: str | None = None,
    ) -> ProjectLabel:
        del scope
        body: dict[str, Any] = {"name": name.strip(), "type": "work_item"}
        if color:
            body["color"] = color.strip()
        if description:
            body["description"] = description.strip()
        raw = await self._transport.request(
            "POST",
            f"/api/v1/projects/{encode_seg(project_id)}/labels",
            json_body=body,
        )
        return _parse_label(unwrap_nested_dict(raw, "label", {"name": name}))

    async def delete_label(
        self, scope: ProjectScope, project_id: str, label_id: str
    ) -> dict[str, Any]:
        del scope
        raw = await self._transport.request(
            "DELETE",
            f"/api/v1/projects/{encode_seg(project_id)}/labels/{encode_seg(label_id)}",
        )
        return raw if isinstance(raw, dict) else {"message": "Label deleted"}

    async def list_activity(
        self,
        scope: ProjectScope,
        *,
        project_id: str | None = None,
        entity_type: str | None = None,
        limit: int = 30,
    ) -> list[ActivityEntry]:
        del scope
        # NOTE: Backend ActivityController uses `@Controller()` with NO `/api/v1/` prefix
        path = (
            f"/api/projects/{encode_seg(project_id)}/activity/feed"
            if project_id and project_id.strip()
            else "/api/activity/feed"
        )
        raw = await self._transport.request(
            "GET",
            path,
            params={
                "limit": max(1, min(limit, 100)),
                "entityType": entity_type.strip() if entity_type else None,
            },
        )
        entries = raw.get("items", []) if isinstance(raw, dict) else raw
        if not isinstance(entries, list):
            return []
        return [_parse_activity(item) for item in entries if isinstance(item, dict)]
