"""Tests for the abstract multi-agent orchestrator, registry, parallel Send, HITL, and SSE."""

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from pydantic import SecretStr

from ai.main import create_app
from ai.orchestrator.context_policy import SlidingWindowContextPolicy
from ai.orchestrator.contracts import (
    AgentOutput,
    AgentRegistry,
    AgentSpec,
    CitationItem,
    DraftOutput,
)
from ai.orchestrator.graph import build_root_graph
from ai.orchestrator.planner import parse_routing_decision
from ai.orchestrator.prompts import build_supervisor_prompt
from ai.orchestrator.state import RootState
from ai.shared.config.database import DatabaseConfig, PostgresConfig, QdrantConfig, RedisConfig
from ai.shared.config.llm import LLMConfig
from ai.shared.config.server import AppEnvironment, ServerConfig
from ai.shared.config.settings import Settings
from ai.shared.llm.factory import ChatRunnable


class _StreamingStubChatModel(BaseChatModel):
    """Minimal streaming BaseChatModel stub that emits genuine on_chat_model_stream events."""

    tokens: list[str]

    @property
    def _llm_type(self) -> str:
        return "streaming-stub"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = "".join(self.tokens)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for tok in self.tokens:
            yield ChatGenerationChunk(message=AIMessageChunk(content=tok))

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for tok in self.tokens:
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=tok))
            if run_manager:
                await run_manager.on_llm_new_token(tok, chunk=chunk)
            yield chunk


def _make_stub(text: str, counter: list[int] | None = None) -> ChatRunnable:
    def _fn(_input: LanguageModelInput) -> AIMessage:
        if counter is not None:
            counter.append(1)
        return AIMessage(content=text)

    return RunnableLambda(_fn)


def _test_settings() -> Settings:
    return Settings(
        server=ServerConfig(
            app_env=AppEnvironment.TEST,
            internal_api_key=SecretStr("test-key"),
        ),
        llm=LLMConfig(api_key=SecretStr("test-llm-key")),
        database=DatabaseConfig(
            postgres=PostgresConfig(database_url=SecretStr("")),
            redis=RedisConfig(redis_url=SecretStr("")),
            qdrant=QdrantConfig(qdrant_url=""),
        ),
    )


def _parse_sse(raw_body: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in raw_body.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = ""
        data_line = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line[7:].strip()
            elif line.startswith("data: "):
                data_line = line[6:].strip()
        if event_name and data_line:
            events.append((event_name, json.loads(data_line)))
    return events


def test_agent_registry_validates_names_and_duplicates() -> None:
    async def _noop(state: RootState, config: RunnableConfig) -> str:
        return "ok"

    for invalid in (
        "turn_init",
        "supervisor",
        "round_gate",
        "direct_respond",
        "synthesizer",
        "output_guard",
        "direct",
        "BadName",
    ):
        with pytest.raises(ValueError):
            AgentSpec(name=invalid, description="invalid", handler=_noop)

    spec = AgentSpec(name="research_agent", description="Finds facts", handler=_noop)
    with pytest.raises(ValueError, match="Duplicate"):
        AgentRegistry([spec, spec])


async def test_empty_registry_skips_router_llm() -> None:
    router_calls: list[int] = []
    main_calls: list[int] = []
    graph = build_root_graph(
        llm=_make_stub("direct answer", main_calls),
        router_llm=_make_stub("should not run", router_calls),
        registry=AgentRegistry(),
    )
    result = await graph.ainvoke({"messages": [HumanMessage(content="Hi")]})
    assert result["route"] == "direct"
    assert result["final_response"] == "direct answer"
    assert router_calls == []
    assert len(main_calls) == 1


async def test_intent_hint_fast_path_bypasses_router_and_synthesizer() -> None:
    router_calls: list[int] = []
    synth_calls: list[int] = []

    async def research_handler(state: RootState, config: RunnableConfig) -> AgentOutput:
        return AgentOutput(
            agent="research",
            content=f"Handled: {state.get('active_instruction')}",
            citations=(CitationItem(source_id="D1", title="Paper 1", page=2),),
            is_final=True,
        )

    registry = AgentRegistry(
        [AgentSpec(name="research", description="Reads papers", handler=research_handler)]
    )
    graph = build_root_graph(
        llm=_make_stub("should not synthesize", synth_calls),
        router_llm=_make_stub("should not route", router_calls),
        registry=registry,
    )
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Summarize paper")],
            "intent_hint": "research",
        }
    )
    assert result["route"] == "research"
    assert result["final_response"] == "Handled: Summarize paper"
    assert len(result["citations"]) == 1
    assert result["citations"][0].source_id == "D1"
    assert router_calls == []
    assert synth_calls == []


async def test_parallel_fan_out_and_multi_turn_state_reset() -> None:
    seen_instructions: list[tuple[str, str]] = []

    async def alpha_handler(state: RootState, config: RunnableConfig) -> AgentOutput:
        seen_instructions.append(("alpha", state.get("active_instruction", "")))
        return AgentOutput(
            agent="alpha",
            content="Alpha finding",
            citations=(CitationItem(source_id="A1", title="Source A"),),
        )

    async def beta_handler(state: RootState, config: RunnableConfig) -> AgentOutput:
        seen_instructions.append(("beta", state.get("active_instruction", "")))
        return AgentOutput(
            agent="beta",
            content="Beta finding",
            citations=(CitationItem(source_id="B1", title="Source B"),),
            artifacts={"count": 2},
        )

    registry = AgentRegistry(
        [
            AgentSpec(name="alpha", description="Alpha specialist", handler=alpha_handler),
            AgentSpec(name="beta", description="Beta specialist", handler=beta_handler),
        ]
    )

    turn_counter = [0]

    def _router_fn(_input: LanguageModelInput) -> AIMessage:
        turn_counter[0] += 1
        if turn_counter[0] == 1:
            return AIMessage(
                content=json.dumps(
                    {
                        "mode": "delegate",
                        "targets": [
                            {"agent": "alpha", "instruction": "Do alpha"},
                            {"agent": "beta", "instruction": "Do beta"},
                        ],
                        "reasoning": "both needed",
                    }
                )
            )
        return AIMessage(
            content=json.dumps(
                {
                    "mode": "delegate",
                    "targets": [{"agent": "alpha", "instruction": "Only alpha"}],
                    "reasoning": "single",
                }
            )
        )

    graph = build_root_graph(
        llm=_make_stub("Synthesized output"),
        router_llm=RunnableLambda(_router_fn),
        registry=registry,
        checkpointer=MemorySaver(),
    )
    cfg: RunnableConfig = {"configurable": {"thread_id": "thread-parallel"}}

    turn1 = await graph.ainvoke(
        {"messages": [HumanMessage(content="Run both")]},
        config=cfg,
    )
    assert turn1["route"] == "alpha,beta"
    assert [c.source_id for c in turn1["citations"]] == ["A1", "B1"]
    assert turn1["artifacts"] == {"count": 2}
    assert ("alpha", "Do alpha") in seen_instructions
    assert ("beta", "Do beta") in seen_instructions

    turn2 = await graph.ainvoke(
        {"messages": [HumanMessage(content="Run alpha only")]},
        config=cfg,
    )
    assert turn2["route"] == "alpha"
    assert [c.source_id for c in turn2["citations"]] == ["A1"]
    assert turn2["artifacts"] == {}
    assert len(turn2["agent_outputs"]) == 1


async def test_router_malformed_output_falls_back_to_direct() -> None:
    async def worker(state: RootState, config: RunnableConfig) -> str:
        raise AssertionError("Worker should not run on malformed router fallback")

    registry = AgentRegistry([AgentSpec(name="worker", description="Does work", handler=worker)])
    graph = build_root_graph(
        llm=_make_stub("Fallback direct answer"),
        router_llm=_make_stub("not valid json at all"),
        registry=registry,
    )
    result = await graph.ainvoke({"messages": [HumanMessage(content="Hi")]})
    assert result["route"] == "direct"
    assert result["final_response"] == "Fallback direct answer"


def test_hitl_interrupt_and_resume_via_api() -> None:
    async def ops_handler(state: RootState, config: RunnableConfig) -> AgentOutput:
        decision = interrupt({"action": "delete_task", "task_id": "T-42"})
        if isinstance(decision, dict) and decision.get("approved"):
            return AgentOutput(agent="ops", content="Task T-42 deleted.", is_final=True)
        return AgentOutput(agent="ops", content="Cancelled.", is_final=True)

    registry = AgentRegistry(
        [AgentSpec(name="ops", description="Workspace CRUD operations", handler=ops_handler)]
    )
    router_json = json.dumps(
        {
            "mode": "delegate",
            "targets": [{"agent": "ops", "instruction": "Delete task T-42"}],
            "reasoning": "ops task",
        }
    )
    app = create_app(
        settings=_test_settings(),
        llm=_make_stub("unused"),
        router_llm=_make_stub(router_json),
        registry=registry,
    )
    headers = {"X-Internal-Key": "test-key"}
    with TestClient(app) as client:
        first = client.post(
            "/chat/sync",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "Delete task T-42"}],
                "space_id": "s1",
                "chat_id": "hitl-chat-1",
            },
        )
        assert first.status_code == 200
        first_body = first.json()
        assert first_body["status"] == "interrupted"
        assert first_body["interrupts"] == [{"action": "delete_task", "task_id": "T-42"}]

        resumed = client.post(
            "/chat/resume/sync",
            headers=headers,
            json={
                "decision": {"approved": True},
                "space_id": "s1",
                "chat_id": "hitl-chat-1",
            },
        )
        assert resumed.status_code == 200
        resumed_body = resumed.json()
        assert resumed_body["status"] == "completed"
        assert resumed_body["content"] == "Task T-42 deleted."


def test_sse_stream_isolates_router_tokens_and_emits_lifecycle_events() -> None:
    async def analyst_handler(state: RootState, config: RunnableConfig) -> AgentOutput:
        return AgentOutput(
            agent="analyst",
            content="Key finding",
            citations=(CitationItem(source_id="P1", title="Study 2026"),),
        )

    registry = AgentRegistry(
        [AgentSpec(name="analyst", description="Analyzes data", handler=analyst_handler)]
    )
    router_json = json.dumps(
        {
            "mode": "delegate",
            "targets": [{"agent": "analyst", "instruction": "Analyze query"}],
            "reasoning": "needs analysis",
        }
    )
    router_model = _StreamingStubChatModel(tokens=[router_json])
    synth_model = _StreamingStubChatModel(tokens=["Hello ", "Synthesized"])

    app = create_app(
        settings=_test_settings(),
        llm=synth_model,
        router_llm=router_model,
        registry=registry,
    )

    with TestClient(app) as client:
        response = client.post(
            "/chat",
            headers={"X-Internal-Key": "test-key"},
            json={
                "messages": [{"role": "user", "content": "Analyze this"}],
                "space_id": "s1",
                "chat_id": "sse-chat-1",
            },
        )
    assert response.status_code == 200
    parsed = _parse_sse(response.text)
    event_names = [name for name, _ in parsed]
    assert event_names[0] == "run.started"
    assert "agent.routed" in event_names
    assert "agent.started" in event_names
    assert "agent.finished" in event_names
    assert event_names[-1] == "run.finished"

    deltas = "".join(data["delta"] for name, data in parsed if name == "text.delta")
    assert deltas == "Hello Synthesized"
    assert router_json not in deltas

    finished_data = parsed[-1][1]
    assert finished_data["route"] == "analyst"
    assert finished_data["citations"][0]["source_id"] == "P1"


async def test_multi_round_sequential_chaining_and_max_rounds_cap() -> None:
    seen_prior_in_step2: list[str] = []

    async def reader_handler(state: RootState, config: RunnableConfig) -> AgentOutput:
        return AgentOutput(agent="reader", content="Extracted spec: build 2 endpoints")

    async def writer_handler(state: RootState, config: RunnableConfig) -> AgentOutput:
        prior = [out.content for out in state.get("agent_outputs", [])]
        seen_prior_in_step2.extend(prior)
        return AgentOutput(
            agent="writer",
            content=f"Created tasks from: {prior[0]}",
            is_final=True,
        )

    registry = AgentRegistry(
        [
            AgentSpec(name="reader", description="Reads docs", handler=reader_handler),
            AgentSpec(name="writer", description="Creates tasks", handler=writer_handler),
        ]
    )

    router_calls = [0]

    def _router_fn(_input: LanguageModelInput) -> AIMessage:
        router_calls[0] += 1
        if router_calls[0] == 1:
            return AIMessage(
                content=json.dumps(
                    {
                        "mode": "delegate",
                        "targets": [{"agent": "reader", "instruction": "Read uploaded doc"}],
                        "needs_followup": True,
                        "plan_summary": "1) reader -> 2) writer",
                        "reasoning": "reader first",
                    }
                )
            )
        return AIMessage(
            content=json.dumps(
                {
                    "mode": "delegate",
                    "targets": [{"agent": "writer", "instruction": "Create tasks"}],
                    "needs_followup": True,  # Even if router keeps asking, max_rounds=2 caps it
                    "reasoning": "writer second",
                }
            )
        )

    synth_inputs: list[str] = []

    def _synth_fn(prompt_input: LanguageModelInput) -> AIMessage:
        synth_inputs.append(str(prompt_input))
        return AIMessage(content="Final multi-round synthesis")

    graph = build_root_graph(
        llm=RunnableLambda(_synth_fn),
        router_llm=RunnableLambda(_router_fn),
        registry=registry,
        max_rounds=2,
    )
    result = await graph.ainvoke({"messages": [HumanMessage(content="Read and create tasks")]})
    assert router_calls[0] == 2
    assert result["round_count"] == 2
    assert result["route"] == "reader->writer"
    assert result["plan_summary"] == "1) reader -> 2) writer"
    assert seen_prior_in_step2 == ["Extracted spec: build 2 endpoints"]
    assert result["final_response"] == "Final multi-round synthesis"
    assert "Extracted spec: build 2 endpoints" in synth_inputs[0]
    assert "Created tasks from: Extracted spec: build 2 endpoints" in synth_inputs[0]


async def test_bulkhead_isolates_failing_parallel_agent_while_preserving_healthy_branch() -> None:
    async def healthy_handler(state: RootState, config: RunnableConfig) -> AgentOutput:
        return AgentOutput(
            agent="healthy",
            content="Healthy data ready",
            citations=(CitationItem(source_id="H1", title="Healthy Ref"),),
        )

    async def broken_handler(state: RootState, config: RunnableConfig) -> AgentOutput:
        raise RuntimeError("Upstream 502 Bad Gateway")

    registry = AgentRegistry(
        [
            AgentSpec(name="healthy", description="Healthy source", handler=healthy_handler),
            AgentSpec(name="broken", description="Flaky source", handler=broken_handler),
        ]
    )

    captured_prompt: list[str] = []

    def _synth_fn(prompt_input: LanguageModelInput) -> AIMessage:
        captured_prompt.append(str(prompt_input))
        return AIMessage(content="Partial synthesis with healthy data")

    router_json = json.dumps(
        {
            "mode": "delegate",
            "targets": [
                {"agent": "healthy", "instruction": "Fetch healthy"},
                {"agent": "broken", "instruction": "Fetch broken"},
            ],
            "reasoning": "parallel check",
        }
    )
    graph = build_root_graph(
        llm=RunnableLambda(_synth_fn),
        router_llm=_make_stub(router_json),
        registry=registry,
    )
    result = await graph.ainvoke({"messages": [HumanMessage(content="Fetch both")]})
    assert result["final_response"] == "Partial synthesis with healthy data"
    assert [c.source_id for c in result["citations"]] == ["H1"]
    outputs_by_agent = {out.agent: out for out in result["agent_outputs"]}
    assert outputs_by_agent["healthy"].status == "ok"
    assert outputs_by_agent["broken"].status == "error"
    assert "RuntimeError: Upstream 502 Bad Gateway" in (outputs_by_agent["broken"].error or "")
    assert "[ERROR]" in captured_prompt[0]


async def test_sliding_window_context_policy_bounds_long_history() -> None:
    seen_message_counts: list[int] = []

    def _counting_llm(prompt_input: LanguageModelInput) -> AIMessage:
        assert isinstance(prompt_input, list)
        seen_message_counts.append(len(prompt_input))
        return AIMessage(content="bounded-ok")

    policy = SlidingWindowContextPolicy(max_messages=4, max_chars=500)
    graph = build_root_graph(
        llm=RunnableLambda(_counting_llm),
        registry=AgentRegistry(),
        context_policy=policy,
    )
    long_history = [HumanMessage(content=f"turn {i}") for i in range(25)]
    result = await graph.ainvoke({"messages": long_history})
    assert result["final_response"] == "bounded-ok"
    assert seen_message_counts == [4]

    oversized = policy.select_messages([HumanMessage(content="x" * 1000)])
    assert len(oversized) == 1
    assert len(str(oversized[0].content)) <= 500


async def test_output_guard_pipeline_transforms_and_filters_citations() -> None:
    async def rag_handler(state: RootState, config: RunnableConfig) -> AgentOutput:
        return AgentOutput(
            agent="rag",
            content="Claim [D1] and hallucinated [FAKE]",
            citations=(CitationItem(source_id="D1", title="Verified Paper"),),
            is_final=True,
        )

    class _CitationStripGuard:
        async def validate(
            self,
            draft: DraftOutput,
            state: Any,
            config: RunnableConfig,
        ) -> DraftOutput:
            cleaned = draft.content.replace(" and hallucinated [FAKE]", "")
            return draft.model_copy(update={"content": f"{cleaned} [verified]"})

    registry = AgentRegistry(
        [AgentSpec(name="rag", description="Document QA", handler=rag_handler)]
    )
    graph = build_root_graph(
        llm=_make_stub("unused"),
        registry=registry,
        output_guards=[_CitationStripGuard()],
    )
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Check paper")],
            "intent_hint": "rag",
        }
    )
    assert result["final_response"] == "Claim [D1] [verified]"
    assert [c.source_id for c in result["citations"]] == ["D1"]


def test_supervisor_prompt_surfaces_context_and_planner_strips_think_blocks() -> None:
    prompt = build_supervisor_prompt(
        "- **rag_agent**: Docs\n- **web_search_agent**: Web",
        request_context={
            "document_ids": ["doc-alpha"],
            "allowed_domains": ["fastapi.tiangolo.com"],
        },
    )
    assert "doc-alpha" in prompt
    assert "rag_agent" in prompt
    assert "fastapi.tiangolo.com" in prompt
    assert "web_search_agent" in prompt

    async def _noop(state: RootState, config: RunnableConfig) -> str:
        del state, config
        return "ok"

    registry = AgentRegistry(
        [
            AgentSpec(name="rag_agent", description="Docs", handler=_noop),
            AgentSpec(name="web_search_agent", description="Web", handler=_noop),
        ]
    )
    decision = parse_routing_decision(
        '<think>Analyze query and context first.</think>\n'
        '{"mode": "delegate", "targets": [{"agent": "rag_agent", "instruction": "Inspect doc"}]}',
        registry,
    )
    assert decision.mode == "delegate"
    assert [t.agent for t in decision.targets] == ["rag_agent"]
