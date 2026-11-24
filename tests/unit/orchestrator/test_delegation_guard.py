"""Unit tests for ADR-001 Context Spine delegation guard in `RootGraphNodes.supervisor`."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from ai.orchestrator.context_policy import SlidingWindowContextPolicy
from ai.orchestrator.contracts import (
    AgentRegistry,
    AgentSpec,
    AgentTaskAssignment,
    RoutePlanner,
    RoutingDecision,
)
from ai.orchestrator.nodes import RootGraphNodes
from ai.orchestrator.state import RootState
from ai.shared.contracts.auth import bind_delegation_token
from ai.shared.contracts.run_control import RunUsage


class _FixedPlanner(RoutePlanner):
    def __init__(self, decision: RoutingDecision) -> None:
        self._decision = decision

    async def plan(
        self,
        state: RootState,
        registry: AgentRegistry,
        config: RunnableConfig,
    ) -> tuple[RoutingDecision, RunUsage]:
        del state, registry, config
        return self._decision, RunUsage()


def _build_nodes(
    *,
    planner: RoutePlanner,
    agent_names: tuple[str, ...],
) -> RootGraphNodes:
    async def _noop(state: RootState, config: RunnableConfig) -> str:
        del state, config
        return "ok"

    stub_llm = RunnableLambda(lambda _inp: AIMessage(content="ok"))
    registry = AgentRegistry(
        [AgentSpec(name=name, description=f"Spec {name}", handler=_noop) for name in agent_names]
    )
    return RootGraphNodes(
        llm=stub_llm,
        registry=registry,
        planner=planner,
        context_policy=SlidingWindowContextPolicy(),
        output_guards=(),
        has_blocking_guards=False,
        max_rounds=2,
    )


async def test_delegation_guard_drops_auth_specialists_when_unauthenticated() -> None:
    nodes = _build_nodes(
        planner=_FixedPlanner(
            RoutingDecision(
                mode="delegate",
                targets=(
                    AgentTaskAssignment(agent="rag_agent", instruction="Search docs"),
                    AgentTaskAssignment(agent="project_agent", instruction="List projects"),
                ),
                needs_followup=False,
            )
        ),
        agent_names=("rag_agent", "project_agent", "library_agent"),
    )
    state: RootState = {
        "messages": [HumanMessage(content="Ai đang phụ trách Citation Viewer UI?")],
        "round_count": 0,
    }
    res = await nodes.supervisor(state, {"configurable": {"space_id": "sp-1"}})
    assert res["route"] == "rag_agent"
    assert [t.agent for t in res["routing_decision"].targets] == ["rag_agent"]


async def test_delegation_guard_falls_back_to_rag_or_direct_when_only_auth_agents() -> None:
    planner = _FixedPlanner(
        RoutingDecision(
            mode="delegate",
            targets=(AgentTaskAssignment(agent="library_agent", instruction="Search items"),),
            needs_followup=True,
        )
    )
    nodes_with_rag = _build_nodes(
        planner=planner,
        agent_names=("rag_agent", "library_agent"),
    )
    state: RootState = {
        "messages": [HumanMessage(content="Tài liệu Helios-AI đạt accuracy mấy?")],
        "round_count": 0,
    }
    res_rag = await nodes_with_rag.supervisor(state, {"configurable": {"space_id": "sp-1"}})
    assert res_rag["route"] == "rag_agent"
    assert res_rag["routing_decision"].needs_followup is False

    nodes_no_rag = _build_nodes(
        planner=planner,
        agent_names=("library_agent",),
    )
    res_direct = await nodes_no_rag.supervisor(state, {"configurable": {"space_id": "sp-1"}})
    assert res_direct["route"] == "direct"
    assert res_direct["routing_decision"].mode == "direct"


async def test_delegation_policy_preserves_targets_with_token_and_guards_intent_hint() -> None:
    nodes = _build_nodes(
        planner=_FixedPlanner(
            RoutingDecision(
                mode="delegate",
                targets=(AgentTaskAssignment(agent="project_agent", instruction="List tasks"),),
                needs_followup=False,
            )
        ),
        agent_names=("rag_agent", "project_agent", "library_agent"),
    )
    state: RootState = {
        "messages": [HumanMessage(content="Liệt kê work items")],
        "round_count": 0,
    }
    with bind_delegation_token("Bearer user-jwt"):
        res_auth = await nodes.supervisor(state, {"configurable": {"space_id": "sp-1"}})
        res_hint_auth = await nodes.supervisor(
            {**state, "intent_hint": "library_agent"},
            {"configurable": {"space_id": "sp-1"}},
        )
    assert res_auth["route"] == "project_agent"
    assert res_hint_auth["route"] == "library_agent"

    # Without a delegation token, intent_hint must still pass through DelegationTokenPolicy.
    res_hint_unauth = await nodes.supervisor(
        {**state, "intent_hint": "library_agent"},
        {"configurable": {"space_id": "sp-1"}},
    )
    assert res_hint_unauth["route"] == "rag_agent"


async def test_intent_hint_editor_with_attached_pdf_triggers_read_before_write() -> None:
    nodes = _build_nodes(
        planner=_FixedPlanner(RoutingDecision(mode="direct", reasoning="unused")),
        agent_names=("rag_agent", "editor_agent"),
    )
    state: RootState = {
        "messages": [HumanMessage(content="Update section 2 using this PDF")],
        "round_count": 0,
        "intent_hint": "editor_agent",
        "context": {
            "attached_sources": [
                {
                    "kind": "upload",
                    "source_id": "file-pdf-1",
                    "file_id": "file-pdf-1",
                    "title": "Experiment.pdf",
                    "has_pdf": True,
                }
            ]
        },
    }
    res = await nodes.supervisor(state, {"configurable": {"space_id": "sp-1"}})
    assert res["route"] == "rag_agent"
    assert [t.agent for t in res["routing_decision"].targets] == ["rag_agent"]
    assert res["routing_decision"].needs_followup is True
