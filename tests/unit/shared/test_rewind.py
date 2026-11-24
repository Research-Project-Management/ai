"""Unit & endpoint tests for LangGraph Time Travel (`list_thread_checkpoints`, `rewind_thread_state`)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from pydantic import SecretStr

from ai.api.dependencies import get_checkpointer, get_mutation_journal, verify_internal_key
from ai.api.routers import threads
from ai.orchestrator.graph import build_root_graph
from ai.orchestrator.rewind import list_thread_checkpoints, rewind_thread_state
from ai.shared.resilience.journal_store import MemoryJournal, MutationRecord


def _build_test_graph() -> tuple[Any, MemorySaver]:
    saver = MemorySaver()

    def _reply(msgs: Any) -> AIMessage:
        return AIMessage(content=f"reply:{len(msgs)}")

    def _route(_msgs: Any) -> AIMessage:
        return AIMessage(content='{"mode":"direct","targets":[]}')

    graph = build_root_graph(
        llm=RunnableLambda(_reply),
        router_llm=RunnableLambda(_route),
        checkpointer=saver,
    )
    return graph, saver


@pytest.mark.asyncio
async def test_rewind_thread_state_rolls_back_messages_to_turn_1() -> None:
    graph, _saver = _build_test_graph()
    cfg: RunnableConfig = {
        "configurable": {"thread_id": "th-1"},
        "metadata": {"space_id": "sp-1"},
    }

    await graph.ainvoke({"messages": [HumanMessage(content="turn one")]}, config=cfg)
    turn1_head = await graph.aget_state({"configurable": {"thread_id": "th-1"}})
    turn1_ckpt = str((turn1_head.config.get("configurable") or {})["checkpoint_id"])

    await graph.ainvoke({"messages": [HumanMessage(content="turn two")]}, config=cfg)
    turn2_head = await graph.aget_state({"configurable": {"thread_id": "th-1"}})
    assert len(turn2_head.values["messages"]) == 4

    checkpoints = await list_thread_checkpoints(graph, chat_id="th-1", limit=20)
    assert any(c.checkpoint_id == turn1_ckpt for c in checkpoints)
    assert checkpoints[0].message_count == 4

    new_ckpt, msg_count = await rewind_thread_state(
        graph,
        chat_id="th-1",
        checkpoint_id=turn1_ckpt,
        space_id="sp-1",
    )
    assert new_ckpt
    assert msg_count == 2

    rewound_head = await graph.aget_state({"configurable": {"thread_id": "th-1"}})
    assert len(rewound_head.values["messages"]) == 2
    assert rewound_head.values["messages"][0].content == "turn one"


@pytest.mark.asyncio
async def test_threads_checkpoints_and_rewind_endpoints_with_saga_undo() -> None:
    graph, saver = _build_test_graph()
    cfg: RunnableConfig = {
        "configurable": {"thread_id": "th-ep"},
        "metadata": {"space_id": "sp-ep"},
    }

    await graph.ainvoke({"messages": [HumanMessage(content="first question")]}, config=cfg)
    snap1 = await graph.aget_state({"configurable": {"thread_id": "th-ep"}})
    ckpt1 = str((snap1.config.get("configurable") or {})["checkpoint_id"])

    await graph.ainvoke({"messages": [HumanMessage(content="create bad tasks")]}, config=cfg)

    journal = MemoryJournal()
    journal.record(
        "sp-ep",
        "tr-t2",
        MutationRecord(
            tool="create_work_item",
            entity_type="work_item",
            entity_id="wi-bad",
            label="Bad Task",
            undo_method="DELETE",
            undo_path="/api/v1/work-items/wi-bad",
        ),
    )

    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        return httpx.Response(200, json={"ok": True})

    app = FastAPI()
    app.include_router(threads.router)
    app.dependency_overrides[verify_internal_key] = lambda: None
    app.dependency_overrides[get_checkpointer] = lambda: saver
    app.dependency_overrides[get_mutation_journal] = lambda: journal
    test_settings = SimpleNamespace(
        project=SimpleNamespace(backend_url="http://backend.local"),
        server=SimpleNamespace(internal_api_key=SecretStr("test-key")),
    )

    with TestClient(app) as client:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as mock_http:
            app.state.container = SimpleNamespace(
                root_graph=graph,
                settings=test_settings,
                shared_http_client=mock_http,
            )
            list_res = client.get("/threads/th-ep/checkpoints?space_id=sp-ep&limit=10")
            assert list_res.status_code == 200
            ckpts = list_res.json()["checkpoints"]
            assert len(ckpts) >= 2

            rw_res = client.post(
                "/threads/th-ep/rewind?space_id=sp-ep",
                headers={"Authorization": "Bearer user-jwt"},
                json={"checkpoint_id": ckpt1, "revert_trace_ids": ["tr-t2"]},
            )
            assert rw_res.status_code == 200
            body = rw_res.json()
            assert body["target_checkpoint_id"] == ckpt1
            assert body["message_count"] == 2
            assert body["reverted_count"] == 1
            assert [r["entity_id"] for r in body["revert_results"]] == ["wi-bad"]
            assert calls == [("DELETE", "/api/v1/work-items/wi-bad")]

            # Verify metadata["space_id"] is preserved after rewind
            state_res = client.get("/threads/th-ep/state?space_id=sp-ep")
            assert state_res.status_code == 200
            assert state_res.json()["message_count"] == 2
