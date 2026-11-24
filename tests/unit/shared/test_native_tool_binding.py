"""Tests for ContextAwareChatRunnable.bind_tools and native run_plan_step."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig

from ai.shared.config.llm import LLMConfig
from ai.shared.llm.factory import ContextAwareChatRunnable
from ai.shared.resilience.observability import LLMCallMetrics
from ai.shared.workflow.plan import run_plan_step


class _FakeToolModel(Runnable[LanguageModelInput, AIMessage]):
    """Minimal chat-model stub supporting bind_tools."""

    def __init__(self, tag: str, responses: Sequence[AIMessage]) -> None:
        super().__init__()
        self.tag = tag
        self._responses = list(responses)
        self.bound_tools_seen: list[tuple[Any, ...]] = []
        self.calls = 0

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        del input, config, kwargs
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        return self.invoke(input, config=config, **kwargs)

    def bind_tools(
        self, tools: Sequence[Any], **kwargs: Any
    ) -> Runnable[LanguageModelInput, AIMessage]:
        del kwargs
        self.bound_tools_seen.append(tuple(tools))
        return self


async def test_context_aware_runnable_forwards_bind_tools_and_rotates_pool() -> None:
    metrics = LLMCallMetrics()
    m1 = _FakeToolModel("m1", [AIMessage(content="from-m1")])
    m2 = _FakeToolModel("m2", [AIMessage(content="from-m2")])
    wrapper = ContextAwareChatRunnable(
        m1,
        config=LLMConfig(model="test-model"),
        role="agent",
        model="test-model",
        metrics=metrics,
        platform_pool=[m1, m2],
    )

    tools = [{"type": "function", "function": {"name": "list_items", "parameters": {}}}]
    bound = wrapper.bind_tools(tools)

    r1 = await bound.ainvoke("first")
    r2 = await bound.ainvoke("second")

    assert r1.content == "from-m1"
    assert r2.content == "from-m2"
    assert len(m1.bound_tools_seen) == 1
    assert len(m2.bound_tools_seen) == 1
    assert metrics.snapshot()["agent:test-model"]["calls"] == 2


async def test_run_plan_step_uses_native_tool_calls_and_direct_final_answer() -> None:
    tool_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "name": "list_items",
                "args": {"limit": 5},
                "type": "tool_call",
            }
        ],
    )
    final_msg = AIMessage(content="Found 2 papers in your library.")
    fake_llm = _FakeToolModel("planner", [tool_msg, final_msg])
    tools = [{"type": "function", "function": {"name": "list_items", "parameters": {}}}]

    step1 = await run_plan_step(
        fake_llm,
        system_prompt="plan",
        user_content="list my papers",
        config={},
        step=1,
        prior_usage=None,
        max_calls=4,
        tools=tools,
    )
    assert step1["done"] is False
    assert step1["pending_calls"] == [{"tool": "list_items", "args": {"limit": 5}, "reason": ""}]

    step2 = await run_plan_step(
        fake_llm,
        system_prompt="plan",
        user_content="observations: [paper A, paper B]",
        config={},
        step=2,
        prior_usage=step1["usage"],
        max_calls=4,
        tools=tools,
    )
    assert step2["done"] is True
    assert step2["pending_calls"] == []
    assert step2["draft_answer"] == "Found 2 papers in your library."


async def test_run_plan_step_supports_prebound_llm_with_native_tools_bound() -> None:
    tool_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "name": "list_items",
                "args": {"limit": 3},
                "type": "tool_call",
            }
        ],
    )
    final_msg = AIMessage(content="Pre-bound final answer.")
    fake_llm = _FakeToolModel("prebound", [tool_msg, final_msg])
    wrapper = ContextAwareChatRunnable(
        fake_llm,
        config=LLMConfig(model="test-model"),
        role="agent",
        model="test-model",
    )
    tools = [{"type": "function", "function": {"name": "list_items", "parameters": {}}}]
    prebound_llm = wrapper.bind_tools(tools)

    step1 = await run_plan_step(
        prebound_llm,
        system_prompt="plan",
        user_content="list my papers",
        config={},
        step=1,
        prior_usage=None,
        max_calls=4,
        native_tools_bound=True,
    )
    assert step1["done"] is False
    assert step1["pending_calls"] == [{"tool": "list_items", "args": {"limit": 3}, "reason": ""}]

    step2 = await run_plan_step(
        prebound_llm,
        system_prompt="plan",
        user_content="observations: [paper A]",
        config={},
        step=2,
        prior_usage=step1["usage"],
        max_calls=4,
        native_tools_bound=True,
    )
    assert step2["done"] is True
    assert step2["draft_answer"] == "Pre-bound final answer."
    # `_bound_cache` reused the single bind_tools() call across both steps.
    assert len(fake_llm.bound_tools_seen) == 1
