"""Pure-DI unit and FastAPI integration tests for the 5-subsystem `project_agent`."""

from __future__ import annotations

import ast
import json
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi.testclient import TestClient
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr

from ai.agents.project.analytics.health_audit import (
    DEFAULT_HEALTH_RULES,
    HealthAuditContext,
    HealthFinding,
    audit_project_health,
)
from ai.agents.project.analytics.widget_builder import (
    build_metric_summary_widget,
    build_work_item_overview_widget,
)
from ai.agents.project.gateway.contracts import (
    ActivityEntry,
    ProjectAnalytics,
    ProjectGatewayError,
    ProjectLabel,
    ProjectMember,
    ProjectOverview,
    ProjectScope,
    ProjectSummary,
    WorkflowState,
    WorkItemComment,
    WorkItemSummary,
)
from ai.agents.project.gateway.http import HttpProjectGateway
from ai.agents.project.module import build_project_module
from ai.agents.project.resolution.resolver import (
    resolve_member,
    resolve_project,
    resolve_workflow_state,
)
from ai.main import create_app
from ai.orchestrator.contracts import AgentRegistry
from ai.shared.config.database import DatabaseConfig, PostgresConfig, QdrantConfig, RedisConfig
from ai.shared.config.embedding import EmbeddingConfig
from ai.shared.config.llm import LLMConfig
from ai.shared.config.project import ProjectConfig
from ai.shared.config.server import AppEnvironment, ServerConfig
from ai.shared.config.settings import Settings
from ai.shared.contracts.auth import get_delegation_token
from ai.shared.llm.factory import ChatRunnable


def _test_settings(
    *, hitl_policy: Literal["destructive", "all_writes", "readonly"] = "destructive"
) -> Settings:
    return Settings(
        server=ServerConfig(
            app_env=AppEnvironment.TEST,
            internal_api_key=SecretStr("test-key"),
        ),
        llm=LLMConfig(api_key=SecretStr("test-llm-key")),
        embedding=EmbeddingConfig(dimensions=64, collection_name="test_knowledge"),
        project=ProjectConfig(hitl_policy=hitl_policy),
        database=DatabaseConfig(
            postgres=PostgresConfig(database_url=SecretStr("")),
            redis=RedisConfig(redis_url=SecretStr("")),
            qdrant=QdrantConfig(qdrant_url=""),
        ),
    )


def _scripted_llm(outputs: list[str]) -> ChatRunnable:
    idx = {"n": 0}

    def _respond(_messages: LanguageModelInput) -> AIMessage:
        pos = min(idx["n"], len(outputs) - 1)
        idx["n"] += 1
        return AIMessage(content=outputs[pos])

    return RunnableLambda(_respond)


class _ScriptedStreamingChatModel(BaseChatModel):
    step_chunks: list[list[str]]
    call_idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-project-streaming"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        pos = min(self.call_idx, len(self.step_chunks) - 1)
        self.call_idx += 1
        text = "".join(self.step_chunks[pos])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        del messages, stop, run_manager, kwargs
        pos = min(self.call_idx, len(self.step_chunks) - 1)
        self.call_idx += 1
        for piece in self.step_chunks[pos]:
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        del messages, stop, run_manager, kwargs
        pos = min(self.call_idx, len(self.step_chunks) - 1)
        self.call_idx += 1
        for piece in self.step_chunks[pos]:
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))


class InMemoryFakeProjectGateway:
    """In-memory adapter implementing `ProjectGateway` for deterministic tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tokens_seen: list[str] = []
        self.archived_ids: list[str] = []
        self.deleted_ids: list[str] = []
        self.removed_members: list[str] = []

        self.project_rows = [
            ProjectSummary(
                id="proj-1",
                name="Flux Core",
                identifier="FLUX",
                description="Main research workspace",
                priority="high",
                state_name="Execution",
                your_role="owner",
                member_count=2,
            )
        ]
        self.states = [
            WorkflowState(id="st-todo", name="Backlog", group="backlog", sequence=1),
            WorkflowState(id="st-prog", name="In Progress", group="started", sequence=2),
            WorkflowState(id="st-done", name="Done", group="completed", sequence=3),
        ]
        self.members = [
            ProjectMember(
                user_id="usr-minh", name="Minh Nguyen", email="minh@flux.dev", role="owner"
            ),
            ProjectMember(
                user_id="usr-an", name="An Tran", email="an@flux.dev", role="contributor"
            ),
        ]
        self.labels = [
            ProjectLabel(id="lbl-bug", name="Bug", color="#ef4444"),
            ProjectLabel(id="lbl-ai", name="AI", color="#8b5cf6"),
        ]
        self.work_item_rows: dict[str, WorkItemSummary] = {
            "wi-1": WorkItemSummary(
                id="wi-1",
                identifier="FLUX-1",
                title="Fix LaTeX diff anchor edge case",
                project_id="proj-1",
                project_name="Flux Core",
                column_id="st-prog",
                state_name="In Progress",
                state_group="started",
                priority="urgent",
                assignee_id="usr-minh",
                assignee_name="Minh Nguyen",
                due_date="2025-01-01T00:00:00Z",
                is_overdue=True,
                completed=False,
            ),
            "wi-2": WorkItemSummary(
                id="wi-2",
                identifier="FLUX-2",
                title="Write Phase 1 benchmarks",
                project_id="proj-1",
                project_name="Flux Core",
                column_id="st-done",
                state_name="Done",
                state_group="completed",
                priority="medium",
                assignee_id="usr-an",
                assignee_name="An Tran",
                completed=True,
            ),
        }
        self.comments: dict[str, list[WorkItemComment]] = {
            "wi-1": [
                WorkItemComment(
                    id="cmt-1",
                    work_item_id="wi-1",
                    author_id="usr-minh",
                    author_name="Minh Nguyen",
                    content="Checking anchor whitespace fallback.",
                )
            ]
        }

    @property
    def projects(self) -> InMemoryFakeProjectGateway:
        return self

    @property
    def work_items(self) -> InMemoryFakeProjectGateway:
        return self

    @property
    def analytics(self) -> InMemoryFakeProjectGateway:
        return self

    def _record(self, op: str, **kwargs: Any) -> None:
        self.calls.append((op, kwargs))
        self.tokens_seen.append(get_delegation_token())

    async def list_projects(
        self,
        scope: ProjectScope,
        *,
        project_type: Literal["created", "shared", "all"] = "all",
        search: str = "",
        priority: str | None = None,
        is_archived: bool | None = None,
    ) -> list[ProjectSummary]:
        self._record("list_projects", space_id=scope.space_id, project_type=project_type)
        items = list(self.project_rows)
        if search:
            q = search.lower()
            items = [p for p in items if q in p.name.lower() or q in p.identifier.lower()]
        if priority:
            items = [p for p in items if p.priority == priority]
        if is_archived is not None:
            items = [p for p in items if p.is_archived == is_archived]
        return items

    async def get_project(self, scope: ProjectScope, project_id: str) -> ProjectSummary:
        self._record("get_project", project_id=project_id)
        for p in self.project_rows:
            if p.id == project_id or p.identifier == project_id:
                return p
        raise ProjectGatewayError(404, f"Project {project_id} not found")

    async def get_project_overview(self, scope: ProjectScope, project_id: str) -> ProjectOverview:
        self._record("get_project_overview", project_id=project_id)
        items = list(self.work_item_rows.values())
        completed = sum(1 for w in items if w.completed)
        overdue = sum(1 for w in items if w.is_overdue and not w.completed)
        return ProjectOverview(
            project_id=project_id,
            name="Flux Core",
            identifier="FLUX",
            priority="high",
            total_members=len(self.members),
            lead_name="Minh Nguyen",
            total_work_items=len(items),
            completed=completed,
            started=1,
            overdue=overdue,
            completion_percentage=50.0,
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
        self._record("update_project", project_id=project_id, name=name)
        del start_date, target_date
        base = await self.get_project(scope, project_id)
        return base.model_copy(
            update={
                **({"name": name} if name is not None else {}),
                **({"description": description} if description is not None else {}),
                **({"priority": priority} if priority is not None else {}),
                **({"is_archived": is_archived} if is_archived is not None else {}),
            }
        )

    async def list_members(
        self,
        scope: ProjectScope,
        project_id: str,
        *,
        role: str | None = None,
        search: str = "",
        limit: int = 50,
    ) -> list[ProjectMember]:
        self._record("list_members", project_id=project_id, role=role)
        del scope, limit
        items = list(self.members)
        if role:
            items = [m for m in items if m.role == role]
        if search:
            q = search.lower()
            items = [m for m in items if q in m.name.lower() or q in m.email.lower()]
        return items

    async def add_member(
        self,
        scope: ProjectScope,
        project_id: str,
        *,
        user_id: str,
        role: str = "contributor",
    ) -> ProjectMember:
        self._record("add_member", project_id=project_id, user_id=user_id, role=role)
        member = ProjectMember(user_id=user_id, name=f"User {user_id}", role=role)
        self.members.append(member)
        return member

    async def update_member_role(
        self,
        scope: ProjectScope,
        project_id: str,
        user_id: str,
        *,
        role: str,
    ) -> ProjectMember:
        self._record("update_member_role", project_id=project_id, user_id=user_id, role=role)
        del scope
        return ProjectMember(user_id=user_id, name="Updated Member", role=role)

    async def remove_member(
        self, scope: ProjectScope, project_id: str, user_id: str
    ) -> dict[str, Any]:
        self._record("remove_member", project_id=project_id, user_id=user_id)
        del scope
        self.removed_members.append(user_id)
        self.members = [m for m in self.members if m.user_id != user_id]
        return {"message": "Member removed"}

    async def list_workflow_states(
        self, scope: ProjectScope, project_id: str
    ) -> list[WorkflowState]:
        self._record("list_workflow_states", project_id=project_id)
        del scope
        return list(self.states)

    async def list_labels(self, scope: ProjectScope, project_id: str) -> list[ProjectLabel]:
        self._record("list_labels", project_id=project_id)
        del scope
        return list(self.labels)

    async def create_label(
        self,
        scope: ProjectScope,
        project_id: str,
        *,
        name: str,
        color: str | None = None,
        description: str | None = None,
    ) -> ProjectLabel:
        self._record("create_label", project_id=project_id, name=name)
        del scope
        lb = ProjectLabel(
            id=f"lbl-{len(self.labels) + 1}",
            name=name,
            color=color or "#64748b",
            description=description or "",
        )
        self.labels.append(lb)
        return lb

    async def delete_label(
        self, scope: ProjectScope, project_id: str, label_id: str
    ) -> dict[str, Any]:
        self._record("delete_label", project_id=project_id, label_id=label_id)
        del scope
        self.labels = [lb for lb in self.labels if lb.id != label_id]
        return {"message": "Label deleted"}

    async def list_activity(
        self,
        scope: ProjectScope,
        *,
        project_id: str | None = None,
        entity_type: str | None = None,
        limit: int = 30,
    ) -> list[ActivityEntry]:
        self._record("list_activity", project_id=project_id, entity_type=entity_type)
        del scope, limit
        return [
            ActivityEntry(
                id="act-1",
                entity_type="work_item",
                entity_id="wi-1",
                verb="updated",
                actor_name="Minh Nguyen",
                project_name="Flux Core",
            )
        ]

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
        self._record("list_work_items", project_id=project_id, column_id=column_id)
        del scope, label, limit
        items = [w for w in self.work_item_rows.values() if w.project_id == project_id]
        if column_id:
            items = [w for w in items if w.column_id == column_id]
        if state_group:
            items = [w for w in items if w.state_group == state_group]
        if priority:
            items = [w for w in items if w.priority == priority]
        if assignee_id:
            items = [w for w in items if w.assignee_id == assignee_id]
        if completed is not None:
            items = [w for w in items if w.completed == completed]
        if search:
            items = [w for w in items if search.lower() in w.title.lower()]
        return items

    async def get_my_work_items(
        self,
        scope: ProjectScope,
        *,
        project_id: str | None = None,
        relation: Literal["assigned", "created", "subscribed", "all"] = "assigned",
    ) -> list[WorkItemSummary]:
        self._record("get_my_work_items", project_id=project_id, relation=relation)
        del scope
        return [
            w for w in self.work_item_rows.values() if not project_id or w.project_id == project_id
        ]

    async def get_work_item(self, scope: ProjectScope, work_item_id: str) -> WorkItemSummary:
        self._record("get_work_item", work_item_id=work_item_id)
        del scope
        for w in self.work_item_rows.values():
            if w.id == work_item_id or w.identifier == work_item_id:
                return w
        raise ProjectGatewayError(404, f"Work item {work_item_id} not found")

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
        self._record("create_work_item", project_id=project_id, title=title)
        del scope, labels
        st = next((s for s in self.states if s.id == column_id), self.states[0])
        mem = next((m for m in self.members if m.user_id == assignee_id), None)
        item = WorkItemSummary(
            id=f"wi-{len(self.work_item_rows) + 1}",
            identifier=f"FLUX-{len(self.work_item_rows) + 1}",
            title=title,
            description=description or "",
            project_id=project_id,
            project_name="Flux Core",
            column_id=st.id,
            state_name=st.name,
            state_group=st.group,
            priority=priority or "none",
            assignee_id=assignee_id,
            assignee_name=mem.name if mem else "",
            label_ids=tuple(label_ids),
            start_date=start_date,
            due_date=due_date,
            parent_work_item_id=parent_work_item_id,
        )
        self.work_item_rows[item.id] = item
        return item

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
        self._record("update_work_item", work_item_id=work_item_id, column_id=column_id)
        del labels
        existing = await self.get_work_item(scope, work_item_id)
        st = next((s for s in self.states if s.id == column_id), None)
        updated = existing.model_copy(
            update={
                **({"title": title} if title is not None else {}),
                **({"description": description} if description is not None else {}),
                **(
                    {"column_id": st.id, "state_name": st.name, "state_group": st.group}
                    if st
                    else {}
                ),
                **({"assignee_id": None, "assignee_name": ""} if clear_assignee else {}),
                **(
                    {"assignee_id": assignee_id}
                    if assignee_id is not None and not clear_assignee
                    else {}
                ),
                **({"priority": priority} if priority is not None else {}),
                **({"due_date": due_date} if due_date is not None else {}),
                **({"start_date": start_date} if start_date is not None else {}),
                **({"completed": completed} if completed is not None else {}),
                **({"label_ids": tuple(label_ids)} if label_ids is not None else {}),
            }
        )
        self.work_item_rows[existing.id] = updated
        return updated

    async def delete_work_item(self, scope: ProjectScope, work_item_id: str) -> dict[str, Any]:
        self._record("delete_work_item", work_item_id=work_item_id)
        del scope
        self.deleted_ids.append(work_item_id)
        self.work_item_rows.pop(work_item_id, None)
        return {"message": "Work item deleted"}

    async def archive_work_item(self, scope: ProjectScope, work_item_id: str) -> WorkItemSummary:
        self._record("archive_work_item", work_item_id=work_item_id)
        existing = await self.get_work_item(scope, work_item_id)
        self.archived_ids.append(existing.id)
        archived = existing.model_copy(update={"is_archived": True})
        self.work_item_rows[existing.id] = archived
        return archived

    async def assign_work_item(
        self,
        scope: ProjectScope,
        project_id: str,
        work_item_id: str,
        *,
        assignee_id: str | None = None,
    ) -> dict[str, Any]:
        self._record("assign_work_item", project_id=project_id, work_item_id=work_item_id)
        del scope
        return {"newAssigneeId": assignee_id}

    async def list_comments(self, scope: ProjectScope, work_item_id: str) -> list[WorkItemComment]:
        self._record("list_comments", work_item_id=work_item_id)
        del scope
        return list(self.comments.get(work_item_id, []))

    async def add_comment(
        self,
        scope: ProjectScope,
        work_item_id: str,
        *,
        content: str,
        project_id: str | None = None,
    ) -> WorkItemComment:
        self._record("add_comment", work_item_id=work_item_id, project_id=project_id)
        del scope
        comment = WorkItemComment(
            id="cmt-new",
            work_item_id=work_item_id,
            author_id="usr-minh",
            author_name="Minh Nguyen",
            content=content,
        )
        self.comments.setdefault(work_item_id, []).append(comment)
        return comment

    async def get_project_analytics(self, scope: ProjectScope, project_id: str) -> ProjectAnalytics:
        self._record("get_project_analytics", project_id=project_id)
        del scope
        return ProjectAnalytics(
            project_id=project_id,
            total_items=2,
            completed_items=1,
            in_progress_items=1,
            overdue_items=1,
            unassigned_items=0,
            completion_rate=0.5,
            state_groups={"started": 1, "completed": 1},
            priority={"urgent": 1, "medium": 1},
            assignee=[{"userId": "usr-minh", "name": "Minh Nguyen", "count": 1, "overdueCount": 1}],
        )

    async def get_user_overview(self, scope: ProjectScope) -> dict[str, Any]:
        self._record("get_user_overview", space_id=scope.space_id)
        return {"projects": 1, "assignedWorkItems": 1}


def _parse_sse(raw_body: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in raw_body.split("\n\n"):
        lines = [line for line in block.strip().splitlines() if line.strip()]
        if not lines:
            continue
        event_name = ""
        data_raw = ""
        for line in lines:
            if line.startswith("event: "):
                event_name = line[7:].strip()
            elif line.startswith("data: "):
                data_raw = line[6:].strip()
        if event_name and data_raw:
            events.append((event_name, json.loads(data_raw)))
    return events


def test_project_entity_resolution_chain_exact_fuzzy_and_ambiguity() -> None:
    members = [
        ProjectMember(user_id="u1", name="Minh Nguyen", email="minh.nguyen@flux.dev", role="owner"),
        ProjectMember(
            user_id="u2", name="Minh Tran", email="minh.tran@flux.dev", role="contributor"
        ),
        ProjectMember(user_id="u3", name="Alice Chen", email="alice@flux.dev", role="reviewer"),
    ]
    states = [
        WorkflowState(id="s1", name="Todo", group="unstarted"),
        WorkflowState(id="s2", name="In Progress", group="started"),
        WorkflowState(id="s3", name="Done", group="completed"),
    ]
    projects = [
        ProjectSummary(id="p1", name="Quantum Compiler", identifier="QC"),
        ProjectSummary(id="p2", name="Bio Paper", identifier="BIO"),
    ]

    exact = resolve_member("u3", members)
    assert exact.ok and exact.match is not None
    assert exact.match.id == "u3" and exact.match.strategy == "exact_id"

    exact_alias = resolve_member("Alice", members)
    assert exact_alias.ok and exact_alias.match is not None
    assert exact_alias.match.id == "u3" and exact_alias.match.strategy == "exact_name"

    fuzzy = resolve_member("Chen", members)
    assert fuzzy.ok and fuzzy.match is not None
    assert fuzzy.match.id == "u3" and fuzzy.match.strategy == "fuzzy_match"

    ambig = resolve_member("Minh", members)
    assert not ambig.ok and len(ambig.ambiguous) == 2
    assert "Ambiguous member" in (ambig.error or "")

    syn = resolve_workflow_state("doing", states)
    assert syn.ok and syn.match is not None and syn.match.id == "s2"

    proj_ident = resolve_project("BIO", projects)
    assert proj_ident.ok and proj_ident.match is not None and proj_ident.match.id == "p2"


def test_project_health_audit_rules_extensibility_and_widget_builders() -> None:
    items = [
        WorkItemSummary(
            id="w1",
            identifier="FLUX-10",
            title="Blocked kernel",
            priority="urgent",
            is_overdue=True,
            completed=False,
        ),
        WorkItemSummary(
            id="w2",
            identifier="FLUX-11",
            title="Done item",
            priority="low",
            state_name="Done",
            state_group="completed",
            assignee_id="u1",
            assignee_name="Alice",
            completed=True,
        ),
    ]
    analytics = ProjectAnalytics(
        project_id="p1",
        total_items=2,
        completed_items=1,
        overdue_items=1,
        completion_rate=0.5,
    )

    class _NoDescriptionUrgentRule:
        def evaluate(self, ctx: HealthAuditContext) -> tuple[HealthFinding, ...]:
            missing = [w for w in ctx.items if w.priority == "urgent" and not w.description]
            if not missing:
                return ()
            return (
                HealthFinding(
                    severity="low",
                    category="missing_spec",
                    summary=f"{len(missing)} urgent item lacks description.",
                    affected_ids=tuple(w.id for w in missing),
                ),
            )

    report = audit_project_health(
        items,
        analytics,
        project_id="p1",
        rules=(*DEFAULT_HEALTH_RULES, _NoDescriptionUrgentRule()),
    )
    categories = {f.category for f in report.findings}
    assert {"overdue", "unassigned_priority", "missing_spec"} <= categories
    assert report.score < 100 and report.overdue_count == 1

    overview_widget = build_work_item_overview_widget(items, title="Sprint Overview")
    assert overview_widget["type"] == "work_item_overview"
    assert overview_widget["total"] == 2 and overview_widget["done"] == 1
    assert overview_widget["overdue"] == 1

    metric_widget = build_metric_summary_widget(analytics, title="Metrics")
    assert metric_widget["type"] == "metric_summary"
    assert metric_widget["metrics"][3]["value"] == "50%"


async def test_http_project_gateway_exact_nestjs_routes_and_activity_prefix() -> None:
    seen: list[tuple[str, str, dict[str, Any] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] | None = None
        if request.content:
            body = json.loads(request.content.decode("utf-8"))
        seen.append((request.method, request.url.path, body))
        path = request.url.path

        if path == "/api/v1/projects/p1/states" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "states": [{"id": "col-1", "name": "In Progress", "group": "started"}],
                        "columns": [{"id": "col-1", "name": "In Progress", "group": "started"}],
                    },
                },
            )
        if path == "/api/projects/p1/activity/feed" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "items": [
                            {
                                "id": "act-1",
                                "entityType": "work_item",
                                "verb": "created",
                                "actor": {"name": "Alice"},
                            }
                        ]
                    },
                },
            )
        if path == "/api/v1/work-items/wi-9" and request.method == "PUT":
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "workItem": {
                            "id": "wi-9",
                            "identifier": "FLUX-9",
                            "title": "Moved item",
                            "columnId": "col-1",
                        }
                    },
                },
            )
        if path == "/api/v1/projects/p1/analytics" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "totalItems": 10,
                        "completedItems": 7,
                        "inProgressItems": 2,
                        "overdueItems": 1,
                        "unassignedItems": 1,
                        "completionRate": 0.7,
                        "stateGroups": {"completed": 7},
                        "priority": {"high": 3},
                        "assignee": [],
                        "labels": [],
                    },
                },
            )
        return httpx.Response(404, json={"error": {"message": f"Unexpected {path}"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gw = HttpProjectGateway(
            base_url="http://backend.local",
            http_client=client,
            token_provider=lambda: "proj-tok-1",
        )
        scope = ProjectScope(space_id="sp-1", project_id="p1", user_id="u1")

        states = await gw.projects.list_workflow_states(scope, "p1")
        assert len(states) == 1 and states[0].id == "col-1"

        act = await gw.projects.list_activity(scope, project_id="p1")
        assert len(act) == 1 and act[0].actor_name == "Alice"

        moved = await gw.work_items.update_work_item(scope, "wi-9", column_id="col-1")
        assert moved.column_id == "col-1" and moved.identifier == "FLUX-9"

        kpis = await gw.analytics.get_project_analytics(scope, "p1")
        assert kpis.total_items == 10 and kpis.completion_rate == 0.7

    paths_seen = [(m, p) for m, p, _ in seen]
    assert ("GET", "/api/v1/projects/p1/states") in paths_seen
    assert ("GET", "/api/projects/p1/activity/feed") in paths_seen
    assert ("PUT", "/api/v1/work-items/wi-9") in paths_seen


def test_project_agent_streaming_widgets_and_tool_events_via_fastapi() -> None:
    fake_gw = InMemoryFakeProjectGateway()
    router_json = json.dumps(
        {
            "mode": "delegate",
            "targets": [
                {"agent": "project_agent", "instruction": "List work items and audit health"}
            ],
            "needs_followup": False,
        }
    )
    step1_tools = json.dumps(
        {
            "action": "call_tools",
            "tool_calls": [
                {"tool": "list_work_items", "args": {"project_id": "proj-1"}},
                {"tool": "audit_project_health", "args": {"project_id": "proj-1"}},
            ],
        }
    )
    step2_done = json.dumps({"action": "respond", "answer": ""})

    router_stub = _ScriptedStreamingChatModel(step_chunks=[[router_json]])
    agent_stub = _ScriptedStreamingChatModel(
        step_chunks=[
            [step1_tools],
            [step2_done],
            ["Found 2 items ", "and 1 overdue risk."],
        ]
    )

    app = create_app(
        settings=_test_settings(),
        llm=agent_stub,
        router_llm=router_stub,
        project_gateway=fake_gw,
    )
    with TestClient(app) as client:
        res = client.post(
            "/chat",
            headers={"X-Internal-Key": "test-key", "Authorization": "Bearer jwt-pj-stream"},
            json={
                "messages": [{"role": "user", "content": "Show project work items and health"}],
                "space_id": "space-p1",
                "chat_id": "pj-stream-1",
                "context": {"project_id": "proj-1", "user_id": "usr-minh"},
            },
        )
    assert res.status_code == 200
    parsed = _parse_sse(res.text)
    called_tools = [d["tool"] for n, d in parsed if n == "tool.called"]
    completed_tools = [d["tool"] for n, d in parsed if n == "tool.completed"]
    assert "list_work_items" in called_tools
    assert "audit_project_health" in called_tools
    assert "audit_project_health" in completed_tools

    deltas = [d["delta"] for n, d in parsed if n == "text.delta"]
    assert "".join(deltas) == "Found 2 items and 1 overdue risk."

    finished_name, finished_payload = parsed[-1]
    assert finished_name == "run.finished"
    assert finished_payload["route"] == "project_agent"
    widgets = finished_payload["artifacts"]["widgets"]
    assert [w["type"] for w in widgets] == ["work_item_overview", "metric_summary"]
    assert "jwt-pj-stream" in fake_gw.tokens_seen


def test_project_agent_hitl_delete_approve_and_remove_member_reject_via_fastapi() -> None:
    headers = {"X-Internal-Key": "test-key", "Authorization": "Bearer jwt-pj-hitl"}

    approve_gw = InMemoryFakeProjectGateway()
    approve_llm = _scripted_llm(
        [
            json.dumps(
                {
                    "action": "call_tools",
                    "tool_calls": [
                        {
                            "tool": "delete_work_item",
                            "args": {"work_item_id": "wi-1", "mode": "delete"},
                        }
                    ],
                }
            ),
            json.dumps({"action": "respond", "answer": "Work item wi-1 deleted."}),
        ]
    )
    approve_mod = build_project_module(llm=approve_llm, gateway=approve_gw)
    approve_app = create_app(
        settings=_test_settings(hitl_policy="destructive"),
        llm=approve_llm,
        registry=AgentRegistry([approve_mod.agent_spec]),
        project_gateway=approve_gw,
    )
    with TestClient(approve_app) as client:
        first = client.post(
            "/chat/sync",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "Delete wi-1"}],
                "space_id": "s1",
                "chat_id": "pj-hitl-approve",
                "intent_hint": "project_agent",
                "context": {"project_id": "proj-1", "user_id": "usr-minh"},
            },
        )
        assert first.status_code == 200
        b1 = first.json()
        assert b1["status"] == "interrupted"
        intr = b1["interrupts"][0]
        assert intr["tool"] == "delete_work_item"
        assert "Delete work item `wi-1`" in intr["description"]
        assert approve_gw.deleted_ids == []

        resumed = client.post(
            "/chat/resume/sync",
            headers=headers,
            json={
                "decision": [{"decision": "approve", "interruptId": intr["id"]}],
                "space_id": "s1",
                "chat_id": "pj-hitl-approve",
            },
        )
        assert resumed.status_code == 200
        b2 = resumed.json()
        assert b2["status"] == "completed"
        assert approve_gw.deleted_ids == ["wi-1"]

    reject_gw = InMemoryFakeProjectGateway()
    reject_llm = _scripted_llm(
        [
            json.dumps(
                {
                    "action": "call_tools",
                    "tool_calls": [
                        {
                            "tool": "remove_project_member",
                            "args": {"member": "An Tran", "project_id": "proj-1"},
                        }
                    ],
                }
            ),
            "Cancelled removing An Tran per your confirmation.",
        ]
    )
    reject_mod = build_project_module(llm=reject_llm, gateway=reject_gw)
    reject_app = create_app(
        settings=_test_settings(hitl_policy="destructive"),
        llm=reject_llm,
        registry=AgentRegistry([reject_mod.agent_spec]),
        project_gateway=reject_gw,
    )
    with TestClient(reject_app) as client:
        p1 = client.post(
            "/chat/sync",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "Remove An Tran"}],
                "space_id": "s1",
                "chat_id": "pj-hitl-reject",
                "intent_hint": "project_agent",
                "context": {"project_id": "proj-1"},
            },
        )
        intr2 = p1.json()["interrupts"][0]
        assert intr2["tool"] == "remove_project_member"
        p2 = client.post(
            "/chat/resume/sync",
            headers=headers,
            json={
                "decision": [{"decision": "reject", "interruptId": intr2["id"]}],
                "space_id": "s1",
                "chat_id": "pj-hitl-reject",
            },
        )
        assert p2.status_code == 200
        assert p2.json()["status"] == "completed"
        assert "cancelled" in p2.json()["content"].lower()
        assert reject_gw.removed_members == []


def test_default_bootstrap_registers_project_agent_and_slice_isolation() -> None:
    app = create_app(settings=_test_settings(), llm=_scripted_llm(["ok"]))
    with TestClient(app) as client:
        agents_res = client.get("/agents", headers={"X-Internal-Key": "test-key"})
    assert agents_res.status_code == 200
    names = [item["name"] for item in agents_res.json()["agents"]]
    assert "project_agent" in names
    assert "editor_agent" in names

    pj_root = Path("src/ai/agents/project")
    assert (pj_root / "module.py").is_file()
    assert list(pj_root.rglob("__init__.py")) == []

    forbidden = ("ai.agents.rag", "ai.agents.web_search", "ai.agents.library", "ai.agents.editor")
    for py_file in pj_root.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(forbidden), f"{py_file} imports {node.module}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(forbidden), f"{py_file} imports {alias.name}"
