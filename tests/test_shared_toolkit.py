"""Interface-level tests for the shared agent toolkit (planning, gating, HITL, transport) and
the production wiring of the default output guard."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, SecretStr

from ai.main import create_app
from ai.orchestrator.guards import CitationIntegrityGuard
from ai.shared.config.database import DatabaseConfig, PostgresConfig, QdrantConfig, RedisConfig
from ai.shared.config.embedding import EmbeddingConfig
from ai.shared.config.llm import LLMConfig
from ai.shared.config.server import AppEnvironment, ServerConfig
from ai.shared.config.settings import Settings
from ai.shared.contracts.hitl import (
    is_blocked_by_policy,
    parse_hitl_approval,
    requires_confirmation,
)
from ai.shared.gateway.transport import BackendTransport
from ai.shared.llm.output import (
    StreamChannelSplitter,
    extract_reasoning,
    extract_text,
    parse_json_object,
    strip_reasoning,
)
from ai.shared.workflow.batch import BatchTracker, dedupe_citations, observation_of
from ai.shared.workflow.gate import run_policy_gate
from ai.shared.workflow.plan import parse_action_plan
from ai.shared.workflow.registry import ToolRegistry


def _settings() -> Settings:
    return Settings(
        server=ServerConfig(app_env=AppEnvironment.TEST, internal_api_key=SecretStr("test-key")),
        llm=LLMConfig(api_key=SecretStr("k")),
        embedding=EmbeddingConfig(dimensions=64, collection_name="toolkit_test"),
        database=DatabaseConfig(
            postgres=PostgresConfig(database_url=SecretStr("")),
            redis=RedisConfig(redis_url=SecretStr("")),
            qdrant=QdrantConfig(qdrant_url=""),
        ),
    )


# --- production wiring -------------------------------------------------------------------


def test_default_app_wires_citation_guard_and_strips_phantom_markers() -> None:
    def _llm(_input: LanguageModelInput) -> AIMessage:
        return AIMessage(content="Claim [[cite:99]] stays unsupported.")

    app = create_app(settings=_settings(), llm=RunnableLambda(_llm))
    with TestClient(app) as client:
        assert any(isinstance(g, CitationIntegrityGuard) for g in app.state.output_guards)
        res = client.post(
            "/chat/sync",
            headers={"X-Internal-Key": "test-key"},
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "space_id": "s1",
                "intent_hint": "direct",
            },
        )
    assert res.status_code == 200
    assert "[[cite:99]]" not in res.json()["content"]


# --- llm output --------------------------------------------------------------------------


def test_parse_json_object_tolerates_reasoning_and_fences() -> None:
    assert parse_json_object('<think>hmm</think>\n{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 2}\n```') == {"a": 2}
    assert parse_json_object("not json") is None
    assert parse_json_object("[1, 2]") is None
    assert strip_reasoning("<think>x</think> ok ") == "ok"
    assert extract_text([{"text": "a"}, "b", {"x": 1}]) == "ab"


def test_parse_action_plan_caps_calls_and_degrades_to_respond() -> None:
    raw = (
        '{"action": "call_tools", "tool_calls": ['
        '{"tool": "a", "args": {"k": 1}}, {"tool": "b"}, {"tool": "c"}]}'
    )
    plan = parse_action_plan(raw, max_calls=2)
    assert plan.action == "call_tools"
    assert [c.tool for c in plan.tool_calls] == ["a", "b"]

    assert parse_action_plan("garbage", max_calls=4).action == "respond"
    empty = parse_action_plan('{"action": "call_tools", "tool_calls": []}', max_calls=4)
    assert empty.action == "respond"


# --- HITL policy -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tier", "policy", "blocked", "confirm"),
    [
        ("read", "readonly", False, False),
        ("propose", "readonly", True, False),
        ("write", "readonly", True, False),
        ("safe_write", "destructive", False, False),
        ("destructive", "destructive", False, True),
        ("safe_write", "all_writes", False, True),
        ("write", "all_writes", False, True),
        ("propose", "all_writes", False, False),
        ("read", "all_writes", False, False),
    ],
)
def test_hitl_policy_matrix(tier: str, policy: str, blocked: bool, confirm: bool) -> None:
    assert is_blocked_by_policy(tier, policy) is blocked
    assert requires_confirmation(tier, policy) is confirm


def test_parse_hitl_approval_payload_shapes() -> None:
    assert parse_hitl_approval(True) is True
    assert parse_hitl_approval("Approve") is True
    assert parse_hitl_approval({"decisions": [{"type": "reject"}]}) is False
    items = [
        {"decision": "approve", "interruptId": "x"},
        {"decision": "reject", "interruptId": "y"},
    ]
    assert parse_hitl_approval(items, interrupt_id="x") is True
    assert parse_hitl_approval(items, interrupt_id="y") is False
    assert parse_hitl_approval([]) is False


# --- gate / registry / batch -------------------------------------------------------------


class _Args(BaseModel):
    item_id: str


class _Spec(BaseModel):
    name: str
    description: str = "demo tool"
    risk_tier: str = "read"
    args_schema: type[BaseModel] = _Args


class _DemoRegistry(ToolRegistry[_Spec]):
    label = "demo"


def _gate(registry: _DemoRegistry, pending: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    return run_policy_gate(
        pending=pending,
        observations=[],
        get_spec=registry.get,
        policy=policy,
        step=1,
        id_prefix="t",
        build_confirmation=lambda spec, iid, args: {"id": iid, "tool": spec.name, "args": args},
    )


def test_policy_gate_validates_blocks_and_approves() -> None:
    registry = _DemoRegistry([_Spec(name="look"), _Spec(name="wipe", risk_tier="destructive")])
    out = _gate(
        registry,
        [
            {"tool": "nope", "args": {}},
            {"tool": "look", "args": {}},
            {"tool": "look", "args": {"item_id": "a"}},
            {"tool": "wipe", "args": {"item_id": "b"}},
        ],
        policy="readonly",
    )
    statuses = [(o["tool"], o["status"]) for o in out["observations"]]
    assert statuses == [("nope", "error"), ("look", "error"), ("wipe", "error")]
    assert out["approved_calls"] == [{"tool": "look", "args": {"item_id": "a"}}]
    assert out["done"] is False


def test_registry_rejects_duplicates_and_renders_catalog() -> None:
    with pytest.raises(ValueError, match="Duplicate demo tool: look"):
        _DemoRegistry([_Spec(name="look"), _Spec(name="look")])
    registry = _DemoRegistry([_Spec(name="look")])
    assert registry.names == frozenset({"look"})
    assert "`look` [read] (item_id*: str): demo tool" in registry.format_catalog()


class _Cite(BaseModel):
    source_id: str
    page: int | None = None
    snippet: str = ""


class _Result(BaseModel):
    tool: str
    status: str
    data: Any = None
    error: str | None = None
    citations: tuple[_Cite, ...] = ()


def test_batch_tracker_terminal_only_for_successful_mutations() -> None:
    ok_write = BatchTracker(expected_calls=2)
    ok_write.observe(risk_tier="write", status="ok")
    ok_write.observe(risk_tier="propose", status="ok")
    assert ok_write.is_terminal is True

    with_read = BatchTracker(expected_calls=2)
    with_read.observe(risk_tier="read", status="ok")
    with_read.observe(risk_tier="write", status="ok")
    assert with_read.is_terminal is False

    failed = BatchTracker(expected_calls=1)
    failed.observe(risk_tier="write", status="error")
    assert failed.is_terminal is False

    assert BatchTracker(expected_calls=0).is_terminal is False
    unknown = BatchTracker(expected_calls=1)
    unknown.observe(risk_tier=None, status=None)
    assert unknown.is_terminal is False


def test_observation_and_citation_dedupe() -> None:
    assert observation_of(_Result(tool="t", status="ok")) == {"tool": "t", "status": "ok"}
    assert observation_of(_Result(tool="t", status="error", error="boom"))["error"] == "boom"
    first = [_Cite(source_id="a", page=1, snippet="x")]
    merged = dedupe_citations(
        first, [_Cite(source_id="a", page=1, snippet="x"), _Cite(source_id="b")]
    )
    assert [c.source_id for c in merged] == ["a", "b"]


# --- transport ---------------------------------------------------------------------------


class _DemoError(Exception):
    def __init__(self, status_code: int | None, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _transport(handler: Any, *, token: str = "tok") -> BackendTransport:
    return BackendTransport(
        base_url="http://backend.local/",
        service="Demo",
        error_factory=_DemoError,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        token_provider=lambda: token,
        internal_api_key="ik",
    )


async def test_transport_unwraps_envelope_and_sends_auth_headers() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["key"] = request.headers["x-internal-key"]
        return httpx.Response(200, json={"success": True, "data": {"ok": 1}})

    assert await _transport(handler).request("GET", "/x") == {"ok": 1}
    assert seen == {"auth": "Bearer tok", "key": "ik"}


async def test_transport_translates_errors_through_factory() -> None:
    def forbidden(_: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": "Forbidden space"})

    with pytest.raises(_DemoError) as denied:
        await _transport(forbidden).request("GET", "/x")
    assert denied.value.status_code == 403
    assert "Forbidden space" in denied.value.detail

    def timeout(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(_DemoError) as slow:
        await _transport(timeout).request("GET", "/x")
    assert slow.value.status_code == 504

    with pytest.raises(_DemoError) as unauth:
        await _transport(forbidden, token="").request("GET", "/x")
    assert unauth.value.status_code == 401


# --- regression tests for review fixes -------------------------------------------------------


def test_latex_escape_is_math_aware() -> None:
    from ai.shared.tabular.builder import _escape_latex_text

    assert _escape_latex_text("Acc_1 ($x_i$) 50%") == r"Acc\_1 ($x_i$) 50\%"
    assert _escape_latex_text(r"Already\_escaped") == r"Already\_escaped"
    assert _escape_latex_text("") == ""


def test_context_facade_tolerates_non_string_task_kind() -> None:
    from ai.shared.context.contracts import EntityDomain, GraphNode
    from ai.shared.context.facade import ResearchContextFacade

    facade = ResearchContextFacade()
    facade.register_node(
        GraphNode(id="g1", domain=EntityDomain.GOVERNANCE, label="T1", attributes={"kind": None})
    )
    facade.register_node(
        GraphNode(id="g2", domain=EntityDomain.GOVERNANCE, label="T2", attributes={"kind": 7})
    )
    from ai.shared.context.contracts import ViewportFocus

    envelope = facade.assemble_context(ViewportFocus(), "status")
    assert envelope.full_prompt


def test_turn_helpers_handle_content_blocks_and_instruction_priority() -> None:
    from langchain_core.messages import HumanMessage

    from ai.shared.workflow.turn import latest_query, recent_tail

    blocks = HumanMessage(content=[{"type": "text", "text": "find RouteLLM"}])
    assert latest_query({"messages": [blocks]}) == "find RouteLLM"
    assert latest_query({"messages": [blocks], "active_instruction": " do X "}) == "do X"
    assert recent_tail([blocks, AIMessage(content="ok")]) == [
        "user: find RouteLLM",
        "assistant: ok",
    ]
    assert latest_query({}) == ""


# --- shared batch execution, metrics, and graph ---------------------------------------------


class _Runner(BaseModel):
    name: str
    risk_tier: str = "write"
    idempotent: bool = False
    args_schema: type[BaseModel] = _Args


class _Events:
    def __init__(self) -> None:
        self.log: list[tuple[str, str]] = []
        self.inputs: list[tuple[str, dict[str, Any] | None]] = []
        self.errors: list[tuple[str, str | None]] = []

    async def emit_tool_started(
        self,
        tool: str,
        config: Any,
        *,
        tool_input: dict[str, Any] | None = None,
    ) -> None:
        self.log.append(("start", tool))
        self.inputs.append((tool, tool_input))

    async def emit_tool_completed(
        self,
        tool: str,
        status: str,
        config: Any,
        *,
        error: str | None = None,
    ) -> None:
        self.log.append((status, tool))
        self.errors.append((tool, error))


class _GatewayBoom(Exception):
    detail = "backend said no"


async def test_run_tool_batch_isolates_failures_and_reports_metrics() -> None:
    from ai.shared.resilience.observability import ToolExecutionMetrics
    from ai.shared.resilience.tool_runner import bind_tool_metrics
    from ai.shared.workflow.execute import run_tool_batch

    specs = {s.name: s for s in (_Runner(name="ok"), _Runner(name="gw"), _Runner(name="crash"))}

    async def invoke(spec: Any, args: BaseModel) -> _Result:
        if spec.name == "gw":
            raise _GatewayBoom
        if spec.name == "crash":
            raise RuntimeError("kaput")
        return _Result(tool=spec.name, status="ok", data={"id": args.model_dump()["item_id"]})

    events = _Events()
    metrics = ToolExecutionMetrics()
    with bind_tool_metrics(metrics):
        outcome = await run_tool_batch(
            approved=[
                {"tool": "ok", "args": {"item_id": "a"}},
                {"tool": "gw", "args": {"item_id": "b"}},
                {"tool": "crash", "args": {"item_id": "c"}},
                {"tool": "missing", "args": {}},
            ],
            get_spec=specs.get,
            invoke=invoke,
            failure=lambda tool, msg: _Result(tool=tool, status="error", error=msg),
            gateway_error=_GatewayBoom,
            events=events,
            config={},
            timeout_seconds=2.0,
            observations=[],
            citations=[],
        )

    by_tool = {o["tool"]: o for o in outcome.observations}
    assert by_tool["ok"]["status"] == "ok"
    assert by_tool["gw"]["error"] == "backend said no"
    assert by_tool["crash"]["error"] == "RuntimeError: kaput"
    assert outcome.is_terminal is False
    assert events.log[:2] == [("start", "ok"), ("ok", "ok")]
    assert ("gw", "backend said no") in events.errors
    assert ("crash", "RuntimeError: kaput") in events.errors
    snap = metrics.snapshot()
    assert snap["ok"]["calls"] == 1 and snap["ok"]["errors"] == 0
    assert snap["crash"]["errors"] == 1


async def test_run_tool_batch_terminal_after_clean_mutations_and_hitl_propagates() -> None:
    from langgraph.errors import GraphBubbleUp

    from ai.shared.workflow.execute import run_tool_batch

    spec = _Runner(name="write")

    async def good(_: Any, args: BaseModel) -> _Result:
        return _Result(tool="write", status="ok")

    common: dict[str, Any] = {
        "get_spec": {"write": spec}.get,
        "failure": lambda tool, msg: _Result(tool=tool, status="error", error=msg),
        "gateway_error": _GatewayBoom,
        "events": _Events(),
        "config": {},
        "timeout_seconds": 2.0,
        "observations": [],
        "citations": [],
    }
    ok = await run_tool_batch(
        approved=[{"tool": "write", "args": {"item_id": "x"}}], invoke=good, **common
    )
    assert ok.is_terminal is True

    async def interrupted(_: Any, args: BaseModel) -> _Result:
        raise GraphBubbleUp()

    with pytest.raises(GraphBubbleUp):
        await run_tool_batch(
            approved=[{"tool": "write", "args": {"item_id": "x"}}], invoke=interrupted, **common
        )


async def test_tool_agent_graph_requires_standard_node_names() -> None:
    from typing_extensions import TypedDict

    from ai.shared.workflow.graph import compile_tool_agent_graph

    class _S(TypedDict, total=False):
        step: int
        done: bool
        pending_calls: list[dict[str, Any]]
        approved_calls: list[dict[str, Any]]
        out: str

    async def plan(state: _S) -> dict[str, Any]:
        return {"step": 1, "pending_calls": [], "done": True}

    async def noop(state: _S) -> dict[str, Any]:
        return {}

    async def synth(state: _S) -> dict[str, Any]:
        return {"out": "done"}

    graph = compile_tool_agent_graph(
        _S,
        {
            "plan_and_act": plan,
            "policy_gate": noop,
            "execute_and_verify": noop,
            "synthesize_response": synth,
        },
        max_steps=3,
    )
    assert (await graph.ainvoke({}))["out"] == "done"


def test_stream_channel_splitter_separates_reasoning_and_text_across_chunks() -> None:
    from langchain_core.messages import AIMessageChunk

    assert (
        extract_text(
            [{"type": "thinking", "thinking": "secret"}, {"type": "text", "text": "shown"}]
        )
        == "shown"
    )
    assert (
        extract_reasoning(
            AIMessageChunk(
                content=[{"type": "reasoning", "text": "plan "}],
                additional_kwargs={"reasoning_content": "step1 "},
            )
        )
        == "step1 plan "
    )

    splitter = StreamChannelSplitter()
    chunks = [
        AIMessageChunk(content="Preface <th"),
        AIMessageChunk(content="ink>inner plan "),
        AIMessageChunk(content="detail</t"),
        AIMessageChunk(content="hink>Final answer x <"),
        AIMessageChunk(content=" 5"),
    ]
    emitted = [pair for chunk in chunks for pair in splitter.feed_chunk(chunk)]
    emitted.extend(splitter.flush())

    text_out = "".join(d for ch, d in emitted if ch == "text")
    reasoning_out = "".join(d for ch, d in emitted if ch == "reasoning")
    assert text_out == "Preface Final answer x < 5"
    assert reasoning_out == "inner plan detail"

    rc_splitter = StreamChannelSplitter()
    rc_events = rc_splitter.feed_chunk(
        AIMessageChunk(content="Visible", additional_kwargs={"reasoning_content": "Deep"})
    )
    rc_events.extend(rc_splitter.flush())
    assert rc_events == [("reasoning", "Deep"), ("text", "Visible")]


async def test_summarize_tool_input_and_run_tool_batch_forwards_summary() -> None:
    from ai.shared.workflow.events import LangGraphToolEvents, NullToolEvents, summarize_tool_input
    from ai.shared.workflow.execute import run_tool_batch

    summary = summarize_tool_input(
        {
            "title": "  " + ("A" * 150),
            "file_path": "main.tex",
            "query": "",
            "content": "\\documentclass{article} ... huge body ...",
            "raw_bibtex": "@article{...}",
        }
    )
    assert summary["file_path"] == "main.tex"
    assert len(summary["title"]) == 120 and summary["title"].endswith("…")
    assert "query" not in summary
    assert "content" not in summary
    assert "raw_bibtex" not in summary

    null_pub = NullToolEvents()
    await null_pub.emit_tool_started("create_page", {}, tool_input={"title": "FFA"})
    lg_pub = LangGraphToolEvents("editor_agent")
    await lg_pub.emit_tool_started("create_page", {}, tool_input={"title": "FFA"})

    events = _Events()
    spec = _Runner(name="ok")

    async def invoke(s: Any, args: BaseModel) -> _Result:
        return _Result(tool=s.name, status="ok", data={"id": args.model_dump()["item_id"]})

    await run_tool_batch(
        approved=[{"tool": "ok", "args": {"item_id": "item-42"}}],
        get_spec={"ok": spec}.get,
        invoke=invoke,
        failure=lambda tool, msg: _Result(tool=tool, status="error", error=msg),
        gateway_error=_GatewayBoom,
        events=events,
        config={},
        timeout_seconds=2.0,
        observations=[],
        citations=[],
    )
    assert events.inputs == [("ok", {"item_id": "item-42"})]


async def test_turn_mutation_journal_lifo_revert_and_endpoint() -> None:
    from ai.shared.contracts.auth import bind_delegation_token
    from ai.shared.resilience.mutations import (
        InMemoryTurnMutationJournal,
        MutationRecord,
        MutationRevertError,
        bind_turn_journal,
        build_revert_artifact,
        record_turn_mutation,
        revert_turn_mutations,
    )

    journal = InMemoryTurnMutationJournal(max_turns=2)
    with bind_turn_journal(journal, space_id="s1", trace_id="t-1"):
        record_turn_mutation(
            MutationRecord(
                tool="create_work_item",
                entity_type="work_item",
                entity_id="wi-1",
                label="Task 1",
                undo_method="DELETE",
                undo_path="/api/v1/work-items/wi-1",
            )
        )
        record_turn_mutation(
            MutationRecord(
                tool="create_page",
                entity_type="page",
                entity_id="pg-1",
                label="FFA",
                undo_method="DELETE",
                undo_path="/api/v1/pages/pages/pg-1",
            )
        )

    art = build_revert_artifact(journal, space_id="s1", trace_id="t-1")
    assert art is not None and art["count"] == 2
    assert [item["entity_id"] for item in art["items"]] == ["wi-1", "pg-1"]

    # Bounded LRU eviction check
    journal.record(
        "s1",
        "t-2",
        MutationRecord(
            tool="create_page",
            entity_type="page",
            entity_id="pg-2",
            label="P2",
            undo_method="DELETE",
            undo_path="/api/v1/pages/pages/pg-2",
        ),
    )
    journal.record(
        "s1",
        "t-3",
        MutationRecord(
            tool="create_page",
            entity_type="page",
            entity_id="pg-3",
            label="P3",
            undo_method="DELETE",
            undo_path="/api/v1/pages/pages/pg-3",
        ),
    )
    assert journal.list_mutations("s1", "t-1") == ()

    # Unit LIFO revert with 404 already_gone and failure isolation
    calls: list[tuple[str, str]] = []

    def fail_handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path.endswith("wi-2"):
            return httpx.Response(500, json={"message": "db error"})
        if request.url.path.endswith("pg-gone"):
            return httpx.Response(404, json={"message": "Not found"})
        return httpx.Response(200, json={"ok": True})

    fail_journal = InMemoryTurnMutationJournal()
    with bind_turn_journal(fail_journal, space_id="s1", trace_id="t-mix"):
        record_turn_mutation(
            MutationRecord(
                tool="create_work_item",
                entity_type="work_item",
                entity_id="wi-2",
                label="Fail Task",
                undo_method="DELETE",
                undo_path="/api/v1/work-items/wi-2",
            )
        )
        record_turn_mutation(
            MutationRecord(
                tool="create_page",
                entity_type="page",
                entity_id="pg-gone",
                label="Deleted Manually",
                undo_method="DELETE",
                undo_path="/api/v1/pages/pages/pg-gone",
            )
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(fail_handler)) as client:
        transport = BackendTransport(
            base_url="http://backend.local",
            service="Revert",
            error_factory=MutationRevertError,
            http_client=client,
            internal_api_key="test-key",
        )
        with bind_delegation_token("Bearer user-jwt"):
            res = await revert_turn_mutations(
                fail_journal, space_id="s1", trace_id="t-mix", transport=transport
            )

    assert [c[1] for c in calls] == ["/api/v1/pages/pages/pg-gone", "/api/v1/work-items/wi-2"]
    assert [r.status for r in res] == ["already_gone", "failed"]
    assert len(fail_journal.list_mutations("s1", "t-mix")) == 2

    # FastAPI endpoint integration test: POST /turns/{trace_id}/revert
    endpoint_calls: list[tuple[str, str, str]] = []

    def ok_handler(request: httpx.Request) -> httpx.Response:
        endpoint_calls.append(
            (request.method, request.url.path, request.headers.get("authorization", ""))
        )
        return httpx.Response(200, json={"ok": True})

    app = create_app(settings=_settings(), llm=RunnableLambda(lambda _: AIMessage(content="ok")))
    with TestClient(app) as client:
        async with httpx.AsyncClient(transport=httpx.MockTransport(ok_handler)) as mock_http:
            app.state.shared_http_client = mock_http
            with bind_turn_journal(app.state.mutation_journal, space_id="s1", trace_id="tr-99"):
                record_turn_mutation(
                    MutationRecord(
                        tool="create_work_item",
                        entity_type="work_item",
                        entity_id="wi-a",
                        label="First",
                        undo_method="DELETE",
                        undo_path="/api/v1/work-items/wi-a",
                    )
                )
                record_turn_mutation(
                    MutationRecord(
                        tool="create_page",
                        entity_type="page",
                        entity_id="pg-b",
                        label="Second",
                        undo_method="DELETE",
                        undo_path="/api/v1/pages/pages/pg-b",
                    )
                )
            resp = client.post(
                "/turns/tr-99/revert?space_id=s1",
                headers={
                    "X-Internal-Key": "test-key",
                    "Authorization": "Bearer user-revert-jwt",
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["reverted_count"] == 2
        assert [r["entity_id"] for r in body["results"]] == ["pg-b", "wi-a"]
        assert [r["status"] for r in body["results"]] == ["reverted", "reverted"]
        assert app.state.mutation_journal.list_mutations("s1", "tr-99") == ()
        assert endpoint_calls == [
            ("DELETE", "/api/v1/pages/pages/pg-b", "Bearer user-revert-jwt"),
            ("DELETE", "/api/v1/work-items/wi-a", "Bearer user-revert-jwt"),
        ]
