"""Unit tests for JournalStore repository/factory, Redis persistence, and selective LIFO revert."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from redis.asyncio import Redis
from redis.exceptions import RedisError

from ai.api.dependencies import get_mutation_journal, verify_internal_key
from ai.api.routers import threads
from ai.shared.contracts.auth import bind_delegation_token
from ai.shared.gateway.transport import BackendTransport
from ai.shared.resilience.journal_store import (
    MemoryJournal,
    MutationRecord,
    RedisJournal,
    build_journal,
)
from ai.shared.resilience.mutations import (
    MutationRevertError,
    bind_turn_journal,
    record_patch_undo,
    revert_turn_mutations,
)


class _FakeRedis:
    def __init__(self, *, fail: bool = False) -> None:
        self.lists: dict[str, list[str]] = {}
        self.ttls: dict[str, int] = {}
        self.fail = fail

    async def rpush(self, key: str, *values: str) -> int:
        if self.fail:
            raise RedisError("redis unavailable")
        bucket = self.lists.setdefault(key, [])
        bucket.extend(values)
        return len(bucket)

    async def expire(self, key: str, ttl: int) -> bool:
        if self.fail:
            raise RedisError("redis unavailable")
        self.ttls[key] = ttl
        return True

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        if self.fail:
            raise RedisError("redis unavailable")
        bucket = self.lists.get(key, [])
        if end == -1:
            return list(bucket[start:])
        return list(bucket[start : end + 1])

    async def delete(self, key: str) -> int:
        if self.fail:
            raise RedisError("redis unavailable")
        return 1 if self.lists.pop(key, None) is not None else 0


def _sample(entity_id: str, label: str, method: str = "DELETE") -> MutationRecord:
    return MutationRecord(
        tool="create_work_item" if method == "DELETE" else "update_work_item",
        entity_type="work_item",
        entity_id=entity_id,
        label=label,
        undo_method=cast(Any, method),
        undo_path=f"/api/v1/work-items/{entity_id}",
    )


def test_build_journal_factory_selects_memory_or_redis() -> None:
    assert isinstance(build_journal(None), MemoryJournal)
    fake = _FakeRedis()
    assert isinstance(build_journal(cast(Redis, fake)), RedisJournal)


def test_memory_journal_lru_remove_and_patch_helper() -> None:
    journal = MemoryJournal(max_turns=2)
    journal.record("s1", "t1", _sample("e1", "First"))
    journal.record("s1", "t1", _sample("e2", "Second"))
    journal.remove("s1", "t1", ["e1"])
    assert [m.entity_id for m in journal.list_mutations("s1", "t1")] == ["e2"]

    with bind_turn_journal(journal, space_id="s1", trace_id="t1"):
        record_patch_undo(
            tool="update_work_item",
            entity_type="work_item",
            entity_id="e3",
            label="Task E3",
            path="/api/v1/work-items/e3",
            before_state={"status": "todo"},
        )
    items = journal.list_mutations("s1", "t1")
    assert items[-1].undo_method == "PATCH"
    assert items[-1].undo_body == {"status": "todo"}

    # Evicts oldest turn when exceeding max_turns=2
    journal.record("s1", "t2", _sample("e4", "T2"))
    journal.record("s1", "t3", _sample("e5", "T3"))
    assert journal.list_mutations("s1", "t1") == ()


@pytest.mark.asyncio
async def test_redis_journal_write_through_persists_across_restarts() -> None:
    fake = _FakeRedis()
    j1 = RedisJournal(cast(Redis, fake), ttl_seconds=3600)
    j1.record("s1", "t1", _sample("wi-1", "Task 1"))
    j1.record("s1", "t1", _sample("pg-1", "Page 1"))

    # Simulate process restart with empty local cache reading from same Redis
    await j1._flush_pending()
    j2 = RedisJournal(cast(Redis, fake), ttl_seconds=3600)
    loaded = await j2.alist("s1", "t1")
    assert [m.entity_id for m in loaded] == ["wi-1", "pg-1"]
    assert fake.ttls["flux:ai:turn_journal:s1:t1"] == 3600

    await j2.aremove("s1", "t1", ["pg-1"])
    remaining = await j2.alist("s1", "t1")
    assert [m.entity_id for m in remaining] == ["wi-1"]

    await j2.aclear("s1", "t1")
    assert await j2.alist("s1", "t1") == ()


@pytest.mark.asyncio
async def test_redis_journal_fails_open_on_redis_error() -> None:
    fake = _FakeRedis(fail=True)
    journal = RedisJournal(cast(Redis, fake))
    journal.record("s1", "t1", _sample("wi-1", "Local fallback"))
    loaded = await journal.alist("s1", "t1")
    assert [m.entity_id for m in loaded] == ["wi-1"]


@pytest.mark.asyncio
async def test_selective_revert_runs_lifo_and_preserves_unselected() -> None:
    fake = _FakeRedis()
    journal = RedisJournal(cast(Redis, fake))
    journal.record("s1", "t-sel", _sample("wi-1", "Task 1"))
    journal.record("s1", "t-sel", _sample("wi-2", "Task 2"))
    journal.record("s1", "t-sel", _sample("pg-1", "Page 1"))

    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = BackendTransport(
            base_url="http://backend.local",
            service="Revert",
            error_factory=MutationRevertError,
            http_client=client,
            internal_api_key="k",
        )
        with bind_delegation_token("Bearer u-jwt"):
            res = await revert_turn_mutations(
                journal,
                space_id="s1",
                trace_id="t-sel",
                transport=transport,
                entity_ids=["wi-1", "pg-1"],
            )

    assert [c[1] for c in calls] == ["/api/v1/work-items/pg-1", "/api/v1/work-items/wi-1"]
    assert [r.status for r in res] == ["reverted", "reverted"]
    # Only wi-2 remains in both local cache and Redis
    remaining = await journal.alist("s1", "t-sel")
    assert [m.entity_id for m in remaining] == ["wi-2"]


@pytest.mark.asyncio
async def test_threads_revert_endpoint_accepts_selective_entity_ids() -> None:
    journal = MemoryJournal()
    journal.record("s1", "tr-sel", _sample("wi-1", "Task 1"))
    journal.record("s1", "tr-sel", _sample("wi-2", "Task 2"))

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"ok": True})

    app = FastAPI()
    app.include_router(threads.router)
    app.dependency_overrides[verify_internal_key] = lambda: None
    app.dependency_overrides[get_mutation_journal] = lambda: journal
    stub_settings = SimpleNamespace(
        project=SimpleNamespace(backend_url="http://backend.local"),
        server=SimpleNamespace(internal_api_key=SecretStr("test-key")),
    )

    with TestClient(app) as client:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            app.state.container = SimpleNamespace(
                settings=stub_settings,
                shared_http_client=mock_http,
            )
            res1 = client.post(
                "/turns/tr-sel/revert?space_id=s1",
                headers={"Authorization": "Bearer u-jwt"},
                json={"entity_ids": ["wi-2"]},
            )
            res2 = client.post(
                "/turns/tr-sel/revert?space_id=s1",
                headers={"Authorization": "Bearer u-jwt"},
            )

    assert res1.status_code == 200
    assert [r["entity_id"] for r in res1.json()["results"]] == ["wi-2"]
    assert res2.status_code == 200
    assert [r["entity_id"] for r in res2.json()["results"]] == ["wi-1"]
    assert calls == ["/api/v1/work-items/wi-2", "/api/v1/work-items/wi-1"]
    assert journal.list_mutations("s1", "tr-sel") == ()


@pytest.mark.asyncio
async def test_project_write_tools_record_update_comment_and_label_undos() -> None:
    from ai.agents.project.gateway.contracts import (
        ProjectGateway,
        ProjectLabel,
        ProjectScope,
        WorkItemComment,
        WorkItemSummary,
    )
    from ai.agents.project.tools.contracts import ProjectToolRuntime
    from ai.agents.project.tools.write_commands import (
        AddCommentArgs,
        CreateLabelArgs,
        UpdateWorkItemArgs,
        handle_add_comment,
        handle_create_label,
        handle_update_work_item,
    )

    before_item = WorkItemSummary(
        id="wi-10",
        title="Old Title",
        description="Old desc",
        project_id="proj-1",
        column_id="col-todo",
        priority="low",
        completed=False,
        assignee_id="usr-1",
    )
    after_item = before_item.model_copy(
        update={"title": "New Title", "priority": "high", "completed": True}
    )

    class _FakeWorkItems:
        async def get_work_item(self, _work_item_id: str) -> WorkItemSummary:
            return before_item

        async def update_work_item(self, _work_item_id: str, **_kwargs: Any) -> WorkItemSummary:
            return after_item

        async def add_comment(
            self, work_item_id: str, *, content: str, **_kwargs: Any
        ) -> WorkItemComment:
            return WorkItemComment(id="cmt-5", work_item_id=work_item_id, content=content)

    class _FakeProjects:
        async def create_label(self, _pid: str, *, name: str, **_kwargs: Any) -> ProjectLabel:
            return ProjectLabel(id="lbl-9", name=name)

    fake_gateway = cast(
        ProjectGateway,
        SimpleNamespace(work_items=_FakeWorkItems(), projects=_FakeProjects()),
    )
    runtime = ProjectToolRuntime(
        gateway=fake_gateway,
        scope=ProjectScope(user_id="usr-1", project_id="proj-1"),
    )

    journal = MemoryJournal()
    with bind_turn_journal(journal, space_id="proj-1", trace_id="tr-proj"):
        await handle_update_work_item(
            runtime,
            UpdateWorkItemArgs(
                work_item_id="wi-10",
                title="New Title",
                priority="high",
                completed=True,
            ),
        )
        await handle_add_comment(
            runtime,
            AddCommentArgs(work_item_id="wi-10", content="Checking progress"),
        )
        await handle_create_label(
            runtime,
            CreateLabelArgs(name="Urgent Fix"),
        )

    muts = journal.list_mutations("proj-1", "tr-proj")
    assert [m.tool for m in muts] == [
        "update_work_item",
        "add_work_item_comment",
        "create_label",
    ]
    assert muts[0].undo_method == "PUT"
    assert muts[0].undo_path == "/api/v1/work-items/wi-10"
    assert muts[0].undo_body == {"title": "Old Title", "priority": "low", "completed": False}
    assert muts[1].undo_method == "DELETE"
    assert muts[1].undo_path == "/api/v1/work-items/comments/cmt-5"
    assert muts[2].undo_method == "DELETE"
    assert muts[2].undo_path == "/api/v1/projects/proj-1/labels/lbl-9"


@pytest.mark.asyncio
async def test_library_write_tools_record_collection_and_note_undos() -> None:
    from ai.agents.library.gateway.contracts import LibraryGateway, LibraryScope
    from ai.agents.library.tools.write import (
        ManageCollectionArgs,
        SaveNoteArgs,
        _handle_manage_collection,
        _handle_save_note,
    )

    class _FakeLibWrite:
        async def manage_collection(self, _scope: LibraryScope, **kwargs: Any) -> dict[str, Any]:
            if kwargs["action"] == "create":
                return {"collection": {"id": "col-7", "name": kwargs.get("name")}}
            return {"ok": True}

        async def save_note(self, _scope: LibraryScope, **kwargs: Any) -> dict[str, Any]:
            return {"id": "note-8", "title": kwargs.get("title") or "Note"}

    gateway = cast(LibraryGateway, SimpleNamespace(write=_FakeLibWrite()))
    project_scope = LibraryScope(user_id="usr-1", space_id="sp-99", scope_type="project")
    personal_scope = LibraryScope(user_id="usr-1", space_id="personal", scope_type="personal")

    journal = MemoryJournal()
    with bind_turn_journal(journal, space_id="sp-99", trace_id="tr-lib"):
        await _handle_manage_collection(
            gateway,
            project_scope,
            ManageCollectionArgs(action="create", name="Survey"),
        )
        await _handle_manage_collection(
            gateway,
            project_scope,
            ManageCollectionArgs(action="assign", collection_id="col-7", item_ids=["it-1", "it-2"]),
        )
        await _handle_manage_collection(
            gateway,
            project_scope,
            ManageCollectionArgs(action="detach", collection_id="col-7", item_ids=["it-1"]),
        )
        await _handle_save_note(
            gateway,
            personal_scope,
            SaveNoteArgs(title="Summary", content_md="Hello"),
        )

    muts = journal.list_mutations("sp-99", "tr-lib")
    assert [m.undo_method for m in muts] == ["DELETE", "POST", "POST", "DELETE"]
    assert muts[0].undo_path == "/api/v1/spaces/sp-99/library/collections/col-7"
    assert muts[1].undo_path == "/api/v1/spaces/sp-99/library/collections/col-7/items/bulk-detach"
    assert muts[1].undo_body == {"itemIds": ["it-1", "it-2"]}
    assert muts[2].undo_path == "/api/v1/spaces/sp-99/library/collections/col-7/items"
    assert muts[2].undo_body == {"itemIds": ["it-1"]}
    assert muts[3].undo_path == "/api/v1/library/notes/note-8"
