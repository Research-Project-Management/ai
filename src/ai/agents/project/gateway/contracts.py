"""Zero-trust Hexagonal Port contracts & normalized DTOs for `project_agent`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ProjectGatewayError(Exception):
    """Normalized HTTP/connection error raised by `ProjectGateway` adapters."""

    def __init__(self, status_code: int | None, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class ProjectScope(BaseModel):
    """Per-request user & project/space context propagated to gateway calls."""

    model_config = ConfigDict(frozen=True)

    space_id: str = ""
    project_id: str | None = None
    user_id: str = ""


class ProjectSummary(BaseModel):
    """Normalized Flux project record."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    identifier: str = ""
    description: str = ""
    priority: str = "none"
    state_name: str = ""
    state_id: str | None = None
    start_date: str | None = None
    target_date: str | None = None
    is_archived: bool = False
    your_role: str = ""
    member_count: int = 0


class ProjectOverview(BaseModel):
    """Comprehensive project overview dashboard snapshot."""

    model_config = ConfigDict(frozen=True)

    project_id: str
    name: str
    identifier: str = ""
    description: str = ""
    priority: str = "none"
    state_name: str = ""
    total_members: int = 0
    lead_name: str = ""
    total_work_items: int = 0
    completed: int = 0
    started: int = 0
    unstarted: int = 0
    backlog: int = 0
    cancelled: int = 0
    overdue: int = 0
    completion_percentage: float = 0.0
    pages_count: int = 0
    library_count: int = 0
    current_update_status: str | None = None
    current_update_message: str | None = None
    recent_activities: list[dict[str, Any]] = Field(default_factory=list)


class ProjectMember(BaseModel):
    """Normalized project member with role and user identity."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    name: str
    email: str = ""
    avatar: str | None = None
    role: str = "contributor"


class WorkflowState(BaseModel):
    """Kanban workflow column/state belonging to work items in a project."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    group: str = "unstarted"
    color: str = ""
    sequence: int = 0
    is_default: bool = False


class ProjectLabel(BaseModel):
    """Work-item label defined in a project."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    color: str = ""
    description: str = ""
    type: str = "work_item"
    parent_id: str | None = None


class WorkItemSummary(BaseModel):
    """Normalized work item record used across tools, analytics, and widgets."""

    model_config = ConfigDict(frozen=True)

    id: str
    identifier: str = ""
    title: str
    description: str = ""
    project_id: str = ""
    project_name: str = ""
    column_id: str | None = None
    state_name: str = ""
    state_group: str = ""
    priority: str = "none"
    completed: bool = False
    is_archived: bool = False
    assignee_id: str | None = None
    assignee_name: str = ""
    assignee_ids: tuple[str, ...] = ()
    assignee_names: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    label_ids: tuple[str, ...] = ()
    start_date: str | None = None
    due_date: str | None = None
    is_overdue: bool = False
    parent_work_item_id: str | None = None
    child_count: int = 0
    created_at: str | None = None
    updated_at: str | None = None


class WorkItemComment(BaseModel):
    """Normalized work-item discussion comment."""

    model_config = ConfigDict(frozen=True)

    id: str
    work_item_id: str = ""
    author_id: str = ""
    author_name: str = ""
    content: str
    created_at: str | None = None


class ProjectAnalytics(BaseModel):
    """Project-level dimensional KPIs, state counts, and assignee workload."""

    model_config = ConfigDict(frozen=True)

    project_id: str = ""
    total_items: int = 0
    completed_items: int = 0
    in_progress_items: int = 0
    overdue_items: int = 0
    unassigned_items: int = 0
    completion_rate: float = 0.0
    state_groups: dict[str, int] = Field(default_factory=dict)
    states: list[dict[str, Any]] = Field(default_factory=list)
    priority: dict[str, int] = Field(default_factory=dict)
    assignee: list[dict[str, Any]] = Field(default_factory=list)
    labels: list[dict[str, Any]] = Field(default_factory=list)


class ActivityEntry(BaseModel):
    """Project or workspace activity feed item."""

    model_config = ConfigDict(frozen=True)

    id: str
    entity_type: str = ""
    entity_id: str = ""
    verb: str = ""
    field: str | None = None
    old_value: str | None = None
    new_value: str | None = None
    actor_name: str = ""
    project_name: str = ""
    created_at: str | None = None


class ProjectsPort(Protocol):
    """Projects, members, workflow states, labels, and activity feed."""

    async def list_projects(
        self,
        scope: ProjectScope,
        *,
        project_type: Literal["created", "shared", "all"] = "all",
        search: str = "",
        priority: str | None = None,
        is_archived: bool | None = None,
    ) -> list[ProjectSummary]: ...

    async def get_project(self, scope: ProjectScope, project_id: str) -> ProjectSummary: ...

    async def get_project_overview(
        self, scope: ProjectScope, project_id: str
    ) -> ProjectOverview: ...

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
    ) -> ProjectSummary: ...

    async def list_members(
        self,
        scope: ProjectScope,
        project_id: str,
        *,
        role: str | None = None,
        search: str = "",
        limit: int = 50,
    ) -> list[ProjectMember]: ...

    async def add_member(
        self,
        scope: ProjectScope,
        project_id: str,
        *,
        user_id: str,
        role: str = "contributor",
    ) -> ProjectMember: ...

    async def update_member_role(
        self,
        scope: ProjectScope,
        project_id: str,
        user_id: str,
        *,
        role: str,
    ) -> ProjectMember: ...

    async def remove_member(
        self, scope: ProjectScope, project_id: str, user_id: str
    ) -> dict[str, Any]: ...

    async def list_workflow_states(
        self, scope: ProjectScope, project_id: str
    ) -> list[WorkflowState]: ...

    async def list_labels(
        self, scope: ProjectScope, project_id: str
    ) -> list[ProjectLabel]: ...

    async def create_label(
        self,
        scope: ProjectScope,
        project_id: str,
        *,
        name: str,
        color: str | None = None,
        description: str | None = None,
    ) -> ProjectLabel: ...

    async def delete_label(
        self, scope: ProjectScope, project_id: str, label_id: str
    ) -> dict[str, Any]: ...

    async def list_activity(
        self,
        scope: ProjectScope,
        *,
        project_id: str | None = None,
        entity_type: str | None = None,
        limit: int = 30,
    ) -> list[ActivityEntry]: ...


class WorkItemsPort(Protocol):
    """Work items (Kanban cards) and their comments."""

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
    ) -> list[WorkItemSummary]: ...

    async def get_my_work_items(
        self,
        scope: ProjectScope,
        *,
        project_id: str | None = None,
        relation: Literal["assigned", "created", "subscribed", "all"] = "assigned",
    ) -> list[WorkItemSummary]: ...

    async def get_work_item(
        self, scope: ProjectScope, work_item_id: str
    ) -> WorkItemSummary: ...

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
    ) -> WorkItemSummary: ...

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
    ) -> WorkItemSummary: ...

    async def delete_work_item(
        self, scope: ProjectScope, work_item_id: str
    ) -> dict[str, Any]: ...

    async def archive_work_item(
        self, scope: ProjectScope, work_item_id: str
    ) -> WorkItemSummary: ...

    async def assign_work_item(
        self,
        scope: ProjectScope,
        project_id: str,
        work_item_id: str,
        *,
        assignee_id: str | None = None,
    ) -> dict[str, Any]: ...

    async def list_comments(
        self, scope: ProjectScope, work_item_id: str
    ) -> list[WorkItemComment]: ...

    async def add_comment(
        self,
        scope: ProjectScope,
        work_item_id: str,
        *,
        content: str,
        project_id: str | None = None,
    ) -> WorkItemComment: ...


class AnalyticsPort(Protocol):
    """Project KPIs and the user's cross-project overview."""

    async def get_project_analytics(
        self, scope: ProjectScope, project_id: str
    ) -> ProjectAnalytics: ...

    async def get_user_overview(self, scope: ProjectScope) -> dict[str, Any]: ...


class ProjectGateway(Protocol):
    """Zero-trust Port between `project_agent` and Flux Project Backend APIs, one port per area."""

    @property
    def projects(self) -> ProjectsPort: ...

    @property
    def work_items(self) -> WorkItemsPort: ...

    @property
    def analytics(self) -> AnalyticsPort: ...
