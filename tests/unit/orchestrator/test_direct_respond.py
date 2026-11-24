"""Unit tests for `RootGraphNodes.direct_respond` system instructions."""

from __future__ import annotations

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from ai.orchestrator.context_policy import SlidingWindowContextPolicy
from ai.orchestrator.contracts import AgentRegistry, RoutePlanner, RoutingDecision
from ai.orchestrator.nodes import RootGraphNodes
from ai.orchestrator.prompts import DIRECT_RESPOND_SYSTEM_PROMPT
from ai.orchestrator.state import RootState


class _NoopPlanner(RoutePlanner):
    async def plan(
        self,
        state: RootState,
        registry: AgentRegistry,
        config: RunnableConfig,
    ) -> RoutingDecision:
        del state, registry, config
        return RoutingDecision(mode="direct", targets=(), needs_followup=False)


async def test_direct_respond_prepends_flux_identity_and_language_system_prompt() -> None:
    """Direct responses must include Flux identity and user-language instructions first."""
    captured: list[list[BaseMessage]] = []

    def _fake_chat(inp: LanguageModelInput) -> AIMessage:
        assert isinstance(inp, list)
        msgs = [m for m in inp if isinstance(m, BaseMessage)]
        assert len(msgs) == len(inp)
        captured.append(msgs)
        return AIMessage(content="Xin chào! Tôi là Flux AI Assistant.")

    nodes = RootGraphNodes(
        llm=RunnableLambda(_fake_chat),
        registry=AgentRegistry(),
        planner=_NoopPlanner(),
        context_policy=SlidingWindowContextPolicy(),
        output_guards=(),
        has_blocking_guards=False,
        max_rounds=2,
    )
    state: RootState = {
        "messages": [HumanMessage(content="xin chào")],
        "round_count": 0,
    }

    result = await nodes.direct_respond(state, {})

    assert result["route"] == "direct"
    assert result["final_response"] == "Xin chào! Tôi là Flux AI Assistant."
    assert len(captured) == 1
    sent = captured[0]
    assert isinstance(sent[0], SystemMessage)
    assert sent[0].content == DIRECT_RESPOND_SYSTEM_PROMPT
    assert "Flux AI Assistant" in sent[0].content
    assert "same language" in sent[0].content
    assert isinstance(sent[1], HumanMessage)
    assert sent[1].content == "xin chào"
