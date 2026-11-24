"""Pure-DI unit and FastAPI integration tests for the vertical-slice `library_agent`."""

from __future__ import annotations

import ast
import json
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr, ValidationError

from ai.agents.library.gateway.contracts import (
    ItemDetail,
    LibraryGatewayError,
    LibraryOverview,
    LibraryScope,
)
from ai.agents.library.gateway.http import HttpLibraryGateway
from ai.agents.library.module import build_library_module
from ai.agents.library.tools.contracts import build_default_tool_registry
from ai.agents.library.workflow.contracts import LibraryAgentConfig
from ai.main import create_app
from ai.shared.config.database import DatabaseConfig, PostgresConfig, QdrantConfig, RedisConfig
from ai.shared.config.embedding import EmbeddingConfig
from ai.shared.config.library import LibraryConfig
from ai.shared.config.llm import LLMConfig
from ai.shared.config.server import AppEnvironment, ServerConfig
from ai.shared.config.settings import Settings
from ai.shared.contracts.auth import bind_delegation_token, get_delegation_token
from ai.shared.contracts.hitl import (
    is_blocked_by_policy,
    parse_hitl_approval,
    requires_confirmation,
)
from ai.shared.llm.factory import ChatRunnable


def _test_settings(*, hitl_policy: str = "destructive") -> Settings:
    return Settings(
        server=ServerConfig(
            app_env=AppEnvironment.TEST,
            internal_api_key=SecretStr("test-key"),
        ),
        llm=LLMConfig(api_key=SecretStr("test-llm-key")),
        embedding=EmbeddingConfig(dimensions=64, collection_name="test_knowledge"),
        library=LibraryConfig(hitl_policy=hitl_policy),  # type: ignore[arg-type]
        database=DatabaseConfig(
            postgres=PostgresConfig(database_url=SecretStr("")),
            redis=RedisConfig(redis_url=SecretStr("")),
            qdrant=QdrantConfig(qdrant_url=""),
        ),
    )


class InMemoryFakeLibraryGateway:
    """In-memory adapter implementing LibraryGateway for deterministic tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tokens_seen: list[str] = []
        self.items: dict[str, dict[str, Any]] = {
            "item-1": {
                "id": "item-1",
                "title": "Attention Is All You Need",
                "authors": ["Ashish Vaswani", "Noam Shazeer"],
                "year": 2017,
                "venue": "NeurIPS",
                "doi": "10.5555/3295222.3295349",
                "tags": ["transformers"],
                "collections": ["Foundations"],
            },
            "item-2": {
                "id": "item-2",
                "title": "BERT Pre-training",
                "authors": ["Jacob Devlin"],
                "year": 2019,
                "venue": "NAACL",
                "doi": "10.18653/v1/N19-1423",
                "tags": ["nlp"],
                "collections": ["Foundations"],
            },
        }
        self.trashed: list[str] = []
        self.merged: list[tuple[str, tuple[str, ...]]] = []

    @property
    def read(self) -> InMemoryFakeLibraryGateway:
        return self

    @property
    def write(self) -> InMemoryFakeLibraryGateway:
        return self

    def _record(self, op: str, **kwargs: Any) -> None:
        self.calls.append((op, kwargs))
        self.tokens_seen.append(get_delegation_token())

    async def search_items(
        self,
        scope: LibraryScope,
        *,
        query: str = "",
        year_from: int | None = None,
        year_to: int | None = None,
        item_type: str | None = None,
        collection_id: str | None = None,
        tag: str | None = None,
        read_status: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        del scope, year_from, year_to, item_type, collection_id, tag, read_status
        self._record("search_items", query=query)
        q = query.strip().lower()
        matched = [item for item in self.items.values() if not q or q in item["title"].lower()][
            :limit
        ]
        return {"items": matched, "returned_count": len(matched), "total_count": len(matched)}

    async def get_overview(self, scope: LibraryScope) -> LibraryOverview:
        del scope
        self._record("get_overview")
        return LibraryOverview(counts={"total": len(self.items)})

    async def get_item(
        self,
        scope: LibraryScope,
        item_id: str,
        *,
        include_notes: bool = True,
        include_relations: bool = True,
    ) -> ItemDetail:
        del scope, include_notes, include_relations
        self._record("get_item", item_id=item_id)
        item = self.items[item_id]
        return ItemDetail(
            id=item["id"], title=item["title"], year=item["year"], venue=item["venue"]
        )

    async def audit_health(
        self,
        scope: LibraryScope,
        *,
        check: str = "all",
    ) -> dict[str, Any]:
        del scope
        self._record("audit_health", check=check)
        return {"duplicates": [{"primaryItemId": "item-1", "duplicateItemIds": ["item-2"]}]}

    async def cite_items(
        self,
        scope: LibraryScope,
        *,
        item_ids: Sequence[str] = (),
        style: str = "apa-7th",
        export_format: str | None = None,
    ) -> dict[str, Any]:
        del scope
        self._record(
            "cite_items", item_ids=list(item_ids), style=style, export_format=export_format
        )
        if export_format == "bibtex":
            return {
                "format": "bibtex",
                "content": "@inproceedings{vaswani2017attention, title={Attention Is All You Need}}",
            }
        return {
            "citations": [
                {"itemId": i, "bibliography": f"Vaswani et al. (2017) [{i}]"} for i in item_ids
            ]
        }

    async def add_paper(
        self,
        scope: LibraryScope,
        *,
        source: str,
        identifier: str = "",
        url: str | None = None,
        collection_id: str | None = None,
    ) -> dict[str, Any]:
        del scope
        self._record(
            "add_paper", source=source, identifier=identifier, url=url, collection_id=collection_id
        )
        return {"runId": "run-add-1", "status": "queued", "identifier": identifier or url}

    async def manage_collection(
        self,
        scope: LibraryScope,
        *,
        action: str,
        collection_id: str | None = None,
        name: str | None = None,
        parent_id: str | None = None,
        item_ids: Sequence[str] = (),
        source_collection_id: str | None = None,
    ) -> dict[str, Any]:
        del scope, parent_id, source_collection_id
        self._record(
            "manage_collection",
            action=action,
            collection_id=collection_id,
            name=name,
            item_ids=list(item_ids),
        )
        return {"ok": True, "action": action}

    async def organize_item(
        self,
        scope: LibraryScope,
        *,
        item_id: str,
        tags: Sequence[str] | None = None,
        read_status: str | None = None,
        is_starred: bool | None = None,
        link_item_ids: Sequence[str] = (),
        unlink_item_id: str | None = None,
    ) -> dict[str, Any]:
        del scope, is_starred, link_item_ids, unlink_item_id
        self._record("organize_item", item_id=item_id, tags=tags, read_status=read_status)
        return {"item_id": item_id, "updated": True}

    async def save_note(
        self,
        scope: LibraryScope,
        *,
        note_id: str | None = None,
        item_id: str | None = None,
        title: str | None = None,
        content_md: str = "",
        tags: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        del scope, note_id, tags
        self._record("save_note", item_id=item_id, title=title, content_md=content_md)
        return {"id": "note-1", "title": title or "Note"}

    async def trash_items(
        self,
        scope: LibraryScope,
        *,
        item_ids: Sequence[str],
    ) -> dict[str, Any]:
        del scope
        self._record("trash_items", item_ids=list(item_ids))
        self.trashed.extend(item_ids)
        return {"deletedCount": len(item_ids), "itemIds": list(item_ids)}

    async def merge_duplicates(
        self,
        scope: LibraryScope,
        *,
        primary_item_id: str,
        duplicate_item_ids: Sequence[str],
    ) -> dict[str, Any]:
        del scope
        self._record(
            "merge_duplicates",
            primary_item_id=primary_item_id,
            duplicate_item_ids=list(duplicate_item_ids),
        )
        self.merged.append((primary_item_id, tuple(duplicate_item_ids)))
        return {"mergedInto": primary_item_id, "mergedCount": len(duplicate_item_ids)}


def _scripted_llm(steps: list[str]) -> ChatRunnable:
    idx = {"n": 0}

    def _respond(_prompt: LanguageModelInput) -> AIMessage:
        pos = min(idx["n"], len(steps) - 1)
        idx["n"] += 1
        return AIMessage(
            content=steps[pos],
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )

    return RunnableLambda(_respond)


def _parse_sse(raw: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in raw.strip().split("\n\n"):
        if not block.strip():
            continue
        ev_name = ""
        ev_data: dict[str, Any] = {}
        for line in block.splitlines():
            if line.startswith("event: "):
                ev_name = line[7:].strip()
            elif line.startswith("data: "):
                ev_data = json.loads(line[6:].strip())
        if ev_name:
            events.append((ev_name, ev_data))
    return events


def test_library_config_defaults_env_prefix_and_url_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIBRARY_ENABLED", "true")
    monkeypatch.setenv("LIBRARY_BACKEND_URL", "http://127.0.0.1:4100/")
    monkeypatch.setenv("LIBRARY_HITL_POLICY", "all_writes")

    cfg = LibraryConfig()
    assert cfg.enabled is True
    assert cfg.backend_url == "http://127.0.0.1:4100"
    assert cfg.hitl_policy == "all_writes"
    assert cfg.max_steps == 5
    assert cfg.max_items_per_call == 20

    with pytest.raises(ValidationError):
        LibraryConfig(backend_url="ftp://invalid-library-host")


def test_bind_delegation_token_contextvar_scoping_and_reset() -> None:
    assert get_delegation_token() == ""
    with bind_delegation_token("Bearer secret-jwt-1"):
        assert get_delegation_token() == "secret-jwt-1"
        with pytest.raises(RuntimeError), bind_delegation_token("inner-jwt-2"):
            assert get_delegation_token() == "inner-jwt-2"
            raise RuntimeError("boom")
        assert get_delegation_token() == "secret-jwt-1"
    assert get_delegation_token() == ""


async def test_http_library_gateway_routes_headers_unwrap_and_errors() -> None:
    recorded: list[tuple[str, str, str, Any]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else None
        recorded.append(
            (
                request.method,
                request.url.path,
                request.headers.get("authorization", ""),
                body,
            )
        )
        if request.url.path == "/api/v1/library/search":
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "items": [
                            {
                                "id": "it-1",
                                "title": "Deep Library Agents",
                                "authors": ["A1", "A2", "A3", "A4"],
                                "year": 2025,
                                "publicationTitle": "ICLR",
                                "doi": "10.1000/test",
                                "tags": [{"tag": {"name": "agents"}}],
                            }
                        ],
                        "meta": {"totalCount": 1},
                    },
                },
            )
        if request.url.path == "/api/v1/spaces/proj-99/library/items/it-1/state":
            return httpx.Response(200, json={"success": True, "data": {"readStatus": "completed"}})
        if request.url.path.endswith("/forbidden"):
            return httpx.Response(403, json={"message": "Forbidden space access"})
        return httpx.Response(404, json={"message": "Not found"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
        gateway = HttpLibraryGateway(
            base_url="http://backend.local",
            max_items_per_call=10,
            http_client=client,
        )

        # Missing token fails cleanly with 401 before sending HTTP
        with pytest.raises(LibraryGatewayError) as no_auth_exc:
            await gateway.read.search_items(
                LibraryScope(space_id="u-1", user_id="u-1"), query="agents"
            )
        assert no_auth_exc.value.status_code == 401

        with bind_delegation_token("jwt-abc-999"):
            res = await gateway.read.search_items(
                LibraryScope(space_id="u-1", user_id="u-1"),
                query="agents",
            )
            assert res["returned_count"] == 1
            assert res["items"][0]["authors"] == ["A1", "A2", "A3"]
            assert res["items"][0]["venue"] == "ICLR"
            assert res["items"][0]["tags"] == ["agents"]

            org = await gateway.write.organize_item(
                LibraryScope(space_id="proj-99", scope_type="project"),
                item_id="it-1",
                read_status="read",
            )
            assert org["state_update"]["readStatus"] == "completed"

            with pytest.raises(LibraryGatewayError) as err_exc:
                await gateway.transport.request("GET", "/api/v1/library/forbidden")
            assert err_exc.value.status_code == 403
            assert "jwt-abc-999" not in str(err_exc.value)

    assert recorded[0][:3] == ("GET", "/api/v1/library/search", "Bearer jwt-abc-999")
    assert recorded[1][0] == "PATCH"
    assert recorded[1][1] == "/api/v1/spaces/proj-99/library/items/it-1/state"
    assert recorded[1][3] == {"readStatus": "completed"}


def test_library_tool_registry_11_tools_and_policies() -> None:
    registry = build_default_tool_registry()
    expected = {
        "search_items": ("read", True),
        "library_stats": ("read", True),
        "get_item": ("read", True),
        "audit_library": ("read", True),
        "cite_items": ("read", True),
        "add_paper": ("safe_write", False),
        "manage_collection": ("safe_write", False),
        "organize_item": ("safe_write", False),
        "save_note": ("safe_write", False),
        "trash_items": ("destructive", False),
        "merge_duplicates": ("destructive", False),
    }
    assert set(registry.names) == set(expected)
    for name, (risk, idemp) in expected.items():
        spec = registry.get(name)
        assert spec is not None
        assert spec.risk_tier == risk
        assert spec.idempotent is idemp

    catalog = registry.format_catalog()
    assert "search_items" in catalog
    assert "trash_items" in catalog

    assert is_blocked_by_policy("read", "readonly") is False
    assert is_blocked_by_policy("safe_write", "readonly") is True
    assert is_blocked_by_policy("destructive", "readonly") is True

    assert requires_confirmation("safe_write", "destructive") is False
    assert requires_confirmation("destructive", "destructive") is True
    assert requires_confirmation("safe_write", "all_writes") is True

    assert parse_hitl_approval(True) is True
    assert parse_hitl_approval("approve") is True
    assert parse_hitl_approval("reject") is False
    assert parse_hitl_approval({"approved": True}) is True
    assert (
        parse_hitl_approval(
            [{"decision": "approve", "interruptId": "lib-2-0-trash_items"}],
            interrupt_id="lib-2-0-trash_items",
        )
        is True
    )
    assert (
        parse_hitl_approval(
            [{"decision": "reject", "interruptId": "lib-2-0-trash_items"}],
            interrupt_id="lib-2-0-trash_items",
        )
        is False
    )
    assert parse_hitl_approval({"decisions": [{"type": "approve"}]}) is True
    assert parse_hitl_approval({"decisions": [{"type": "reject"}]}) is False


def test_library_agent_multi_step_read_and_citation_flow() -> None:
    fake_gw = InMemoryFakeLibraryGateway()
    llm = _scripted_llm(
        [
            json.dumps(
                {
                    "action": "call_tools",
                    "tool_calls": [
                        {
                            "tool": "search_items",
                            "args": {"query": "Attention"},
                            "reason": "find paper",
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "action": "call_tools",
                    "tool_calls": [
                        {
                            "tool": "cite_items",
                            "args": {"item_ids": ["item-1"], "export_format": "bibtex"},
                            "reason": "export bibtex",
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "action": "respond",
                    "answer": "```bibtex\n@inproceedings{vaswani2017attention, title={Attention Is All You Need}}\n```",
                }
            ),
        ]
    )
    app = create_app(
        settings=_test_settings(),
        llm=llm,
        library_gateway=fake_gw,
    )
    with TestClient(app) as client:
        res = client.post(
            "/chat/sync",
            headers={"X-Internal-Key": "test-key", "Authorization": "Bearer user-tok-1"},
            json={
                "messages": [{"role": "user", "content": "Xuất BibTeX bài Attention"}],
                "space_id": "s1",
                "chat_id": "lib-cite-1",
                "intent_hint": "library_agent",
            },
        )
    assert res.status_code == 200
    body = res.json()
    assert body["route"] == "library_agent"
    assert "vaswani2017attention" in body["content"]
    assert body["citations"][0]["source_id"] == "item-1"
    assert [op for op, _ in fake_gw.calls] == ["search_items", "cite_items"]
    assert fake_gw.tokens_seen == ["user-tok-1", "user-tok-1"]


def test_library_agent_safe_write_runs_directly_under_destructive_policy() -> None:
    fake_gw = InMemoryFakeLibraryGateway()
    llm = _scripted_llm(
        [
            json.dumps(
                {
                    "action": "call_tools",
                    "tool_calls": [
                        {
                            "tool": "add_paper",
                            "args": {"source": "arxiv", "identifier": "1706.03762"},
                        },
                        {
                            "tool": "save_note",
                            "args": {"title": "Key Idea", "content_md": "Self-attention"},
                        },
                    ],
                }
            ),
            json.dumps(
                {
                    "action": "respond",
                    "answer": "Đã thêm bài arXiv:1706.03762 và lưu ghi chú.",
                }
            ),
        ]
    )
    from ai.orchestrator.contracts import AgentRegistry

    mod = build_library_module(
        llm=llm,
        gateway=fake_gw,
        config=LibraryConfig(hitl_policy="destructive"),
    )
    app = create_app(
        settings=_test_settings(hitl_policy="destructive"),
        llm=llm,
        registry=AgentRegistry([mod.agent_spec]),
    )
    with TestClient(app) as client:
        res = client.post(
            "/chat/sync",
            headers={"X-Internal-Key": "test-key", "Authorization": "Bearer user-tok-safe-write"},
            json={
                "messages": [{"role": "user", "content": "Thêm arXiv 1706.03762 và lưu note"}],
                "space_id": "s1",
                "chat_id": "lib-safe-write-1",
                "intent_hint": "library_agent",
            },
        )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "completed"
    assert [op for op, _ in fake_gw.calls] == ["add_paper", "save_note"]


def test_library_agent_hitl_trash_approve_and_merge_reject_via_fastapi() -> None:
    fake_gw = InMemoryFakeLibraryGateway()
    approve_llm = _scripted_llm(
        [
            json.dumps(
                {
                    "action": "call_tools",
                    "tool_calls": [{"tool": "search_items", "args": {"query": "Attention"}}],
                }
            ),
            json.dumps(
                {
                    "action": "call_tools",
                    "tool_calls": [{"tool": "trash_items", "args": {"item_ids": ["item-1"]}}],
                }
            ),
            "Đã chuyển bài Attention Is All You Need vào thùng rác.",
        ]
    )
    from ai.orchestrator.contracts import AgentRegistry

    mod = build_library_module(llm=approve_llm, gateway=fake_gw)
    app = create_app(
        settings=_test_settings(),
        llm=approve_llm,
        registry=AgentRegistry([mod.agent_spec]),
    )
    headers = {"X-Internal-Key": "test-key", "Authorization": "Bearer user-tok-hitl"}

    with TestClient(app) as client:
        first = client.post(
            "/chat/sync",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "Xóa bài Attention"}],
                "space_id": "s1",
                "chat_id": "lib-hitl-approve",
                "intent_hint": "library_agent",
            },
        )
        assert first.status_code == 200
        first_body = first.json()
        assert first_body["status"] == "interrupted"
        intr = first_body["interrupts"][0]
        assert intr["tool"] == "trash_items"
        assert "Attention Is All You Need" in intr["description"]
        assert fake_gw.trashed == []
        assert [op for op, _ in fake_gw.calls] == ["search_items"]

        resumed = client.post(
            "/chat/resume/sync",
            headers=headers,
            json={
                "decision": [{"decision": "approve", "interruptId": intr["id"]}],
                "space_id": "s1",
                "chat_id": "lib-hitl-approve",
            },
        )
        assert resumed.status_code == 200
        resumed_body = resumed.json()
        assert resumed_body["status"] == "completed"
        assert "thùng rác" in resumed_body["content"]
        assert fake_gw.trashed == ["item-1"]
        # Child subgraph resumed at policy_gate without re-running search_items
        assert [op for op, _ in fake_gw.calls] == ["search_items", "trash_items"]

    # Reject merge_duplicates
    reject_gw = InMemoryFakeLibraryGateway()
    reject_llm = _scripted_llm(
        [
            json.dumps(
                {
                    "action": "call_tools",
                    "tool_calls": [
                        {
                            "tool": "merge_duplicates",
                            "args": {"primary_item_id": "item-1", "duplicate_item_ids": ["item-2"]},
                        }
                    ],
                }
            ),
            "Thao tác gộp bài trùng đã bị hủy theo yêu cầu của bạn.",
        ]
    )
    reject_mod = build_library_module(llm=reject_llm, gateway=reject_gw)
    reject_app = create_app(
        settings=_test_settings(),
        llm=reject_llm,
        registry=AgentRegistry([reject_mod.agent_spec]),
    )
    with TestClient(reject_app) as client:
        p1 = client.post(
            "/chat/sync",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "Gộp item-2 vào item-1"}],
                "space_id": "s1",
                "chat_id": "lib-hitl-reject",
                "intent_hint": "library_agent",
            },
        )
        assert p1.json()["status"] == "interrupted"
        p2 = client.post(
            "/chat/resume/sync",
            headers=headers,
            json={
                "decision": {"decision": "reject"},
                "space_id": "s1",
                "chat_id": "lib-hitl-reject",
            },
        )
        assert p2.status_code == 200
        assert p2.json()["status"] == "completed"
        assert reject_gw.merged == []
        assert reject_gw.calls == []


def test_library_agent_readonly_policy_blocks_writes() -> None:
    fake_gw = InMemoryFakeLibraryGateway()
    llm = _scripted_llm(
        [
            json.dumps(
                {
                    "action": "call_tools",
                    "tool_calls": [
                        {"tool": "add_paper", "args": {"source": "doi", "identifier": "10.1000/1"}}
                    ],
                }
            ),
            "Không thể thêm bài vì thư viện đang ở chế độ chỉ đọc.",
        ]
    )
    from ai.orchestrator.contracts import AgentRegistry

    mod = build_library_module(
        llm=llm,
        gateway=fake_gw,
        config=LibraryConfig(hitl_policy="readonly"),
    )
    app = create_app(
        settings=_test_settings(hitl_policy="readonly"),
        llm=llm,
        registry=AgentRegistry([mod.agent_spec]),
    )
    with TestClient(app) as client:
        res = client.post(
            "/chat/sync",
            headers={"X-Internal-Key": "test-key", "Authorization": "Bearer user-tok-safe-write"},
            json={
                "messages": [{"role": "user", "content": "Thêm bài DOI 10.1000/1"}],
                "space_id": "s1",
                "chat_id": "lib-ro-1",
                "intent_hint": "library_agent",
            },
        )
    assert res.status_code == 200
    assert res.json()["status"] == "completed"
    assert fake_gw.calls == []


def test_library_agent_sse_stream_preserves_token_and_emits_hitl_interrupted() -> None:
    fake_gw = InMemoryFakeLibraryGateway()
    llm = _scripted_llm(
        [
            json.dumps(
                {
                    "action": "call_tools",
                    "tool_calls": [{"tool": "search_items", "args": {"query": "Attention"}}],
                }
            ),
            json.dumps(
                {
                    "action": "call_tools",
                    "tool_calls": [{"tool": "trash_items", "args": {"item_ids": ["item-1"]}}],
                }
            ),
            json.dumps({"action": "respond", "answer": "Done"}),
        ]
    )
    from ai.orchestrator.contracts import AgentRegistry

    mod = build_library_module(llm=llm, gateway=fake_gw)
    app = create_app(
        settings=_test_settings(),
        llm=llm,
        registry=AgentRegistry([mod.agent_spec]),
    )
    with TestClient(app) as client:
        agents_res = client.get("/agents", headers={"X-Internal-Key": "test-key"})
        assert agents_res.status_code == 200

        sse_res = client.post(
            "/chat",
            headers={"X-Internal-Key": "test-key", "Authorization": "Bearer sse-jwt-999"},
            json={
                "messages": [{"role": "user", "content": "Xóa bài Attention"}],
                "space_id": "s1",
                "chat_id": "lib-sse-1",
                "intent_hint": "library_agent",
            },
        )
    assert sse_res.status_code == 200
    parsed = _parse_sse(sse_res.text)
    event_names = [name for name, _ in parsed]
    assert "hitl.interrupted" in event_names
    assert fake_gw.tokens_seen == ["sse-jwt-999"]


def test_default_bootstrap_registers_library_agent_and_slice_isolation() -> None:
    app = create_app(settings=_test_settings(), llm=_scripted_llm(["ok"]))
    with TestClient(app) as client:
        agents_res = client.get("/agents", headers={"X-Internal-Key": "test-key"})
    assert agents_res.status_code == 200
    names = [item["name"] for item in agents_res.json()["agents"]]
    assert "library_agent" in names
    assert "rag_agent" in names
    assert "web_search_agent" in names

    lib_root = Path("src/ai/agents/library")
    assert (lib_root / "module.py").is_file()
    assert list(lib_root.rglob("__init__.py")) == []

    for py_file in lib_root.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("ai.agents.rag"), (
                    f"{py_file} imports {node.module}"
                )
                assert not node.module.startswith("ai.agents.web_search"), (
                    f"{py_file} imports {node.module}"
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("ai.agents.rag")
                    assert not alias.name.startswith("ai.agents.web_search")


def test_action_plan_sanitization_with_config() -> None:
    cfg = LibraryAgentConfig(max_steps=4, hitl_policy="destructive")
    assert cfg.max_steps == 4


class _ScriptedStreamingChatModel(BaseChatModel):
    step_chunks: list[list[str]]
    call_idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-streaming-stub"

    def _next_chunks(self) -> list[str]:
        pos = min(self.call_idx, len(self.step_chunks) - 1)
        self.call_idx += 1
        return self.step_chunks[pos]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="".join(self._next_chunks())))]
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        del messages, stop, run_manager, kwargs
        for tok in self._next_chunks():
            yield ChatGenerationChunk(message=AIMessageChunk(content=tok))

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        del messages, stop, kwargs
        for tok in self._next_chunks():
            msg_chunk = AIMessageChunk(
                content=tok,
                usage_metadata={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            )
            chunk = ChatGenerationChunk(message=msg_chunk)
            if run_manager:
                await run_manager.on_llm_new_token(tok, chunk=chunk)
            yield chunk


def test_library_agent_sse_streams_incremental_tokens_and_hides_planner_json() -> None:
    fake_gw = InMemoryFakeLibraryGateway()
    plan_search_json = json.dumps(
        {
            "action": "call_tools",
            "tool_calls": [{"tool": "search_items", "args": {"query": "Attention"}}],
        }
    )
    plan_done_json = json.dumps({"action": "respond", "answer": ""})
    router_json = json.dumps(
        {
            "mode": "delegate",
            "targets": [{"agent": "library_agent", "instruction": "Tìm bài Attention"}],
            "needs_followup": False,
        }
    )

    router_stub = _ScriptedStreamingChatModel(step_chunks=[[router_json]])
    agent_stub = _ScriptedStreamingChatModel(
        step_chunks=[
            [plan_search_json],
            [plan_done_json],
            ["Tìm ", "thấy ", "1 bài."],
        ]
    )
    app = create_app(
        settings=_test_settings(),
        llm=agent_stub,
        router_llm=router_stub,
        library_gateway=fake_gw,
    )
    with TestClient(app) as client:
        res = client.post(
            "/chat",
            headers={"X-Internal-Key": "test-key", "Authorization": "Bearer stream-jwt-1"},
            json={
                "messages": [{"role": "user", "content": "Tìm bài Attention"}],
                "space_id": "s1",
                "chat_id": "lib-inc-stream-1",
            },
        )
    assert res.status_code == 200
    parsed = _parse_sse(res.text)
    deltas = [data["delta"] for name, data in parsed if name == "text.delta"]
    assert deltas == ["Tìm ", "thấy ", "1 bài."]
    combined = "".join(deltas)
    assert plan_search_json not in combined
    assert plan_done_json not in combined
    assert router_json not in combined
    assert parsed[-1][0] == "run.finished"
    assert parsed[-1][1]["route"] == "library_agent"


async def test_library_event_publishers() -> None:
    from langchain_core.runnables import RunnableConfig

    from ai.shared.workflow.events import LangGraphToolEvents, NullToolEvents

    null_pub = NullToolEvents()
    await null_pub.emit_tool_started("search_items", RunnableConfig())
    await null_pub.emit_tool_completed("search_items", "ok", RunnableConfig())

    lg_pub = LangGraphToolEvents("library_agent")
    await lg_pub.emit_tool_started("search_items", RunnableConfig())
    await lg_pub.emit_tool_completed("search_items", "ok", RunnableConfig())
