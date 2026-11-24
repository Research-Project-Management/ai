"""Workflow node runner implementing `plan_and_act -> policy_gate -> execute_and_verify -> synthesize_response`."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.runnables import RunnableConfig

from ai.agents.library.gateway.contracts import (
    LibraryGateway,
    LibraryGatewayError,
    LibraryScope,
)
from ai.agents.library.tools.contracts import LibraryToolRegistry, ToolResult
from ai.agents.library.workflow.prompts import (
    build_planner_prompt,
    build_synthesizer_prompt,
    format_observations_block,
)
from ai.agents.library.workflow.state import LibraryState
from ai.shared.llm.factory import ChatRunnable
from ai.shared.workflow.events import LangGraphToolEvents, ToolEventPublisher
from ai.shared.workflow.execute import run_tool_batch
from ai.shared.workflow.gate import run_policy_gate
from ai.shared.workflow.plan import ToolAgentConfig, bind_planner_tools, run_plan_step
from ai.shared.workflow.synthesize import format_conversation_tail, run_synthesize_step


def _collect_known_titles(value: Any, out: dict[str, str]) -> None:
    if isinstance(value, dict):
        item_id = value.get("id")
        title = value.get("title") or value.get("name")
        if (
            isinstance(item_id, str)
            and item_id.strip()
            and isinstance(title, str)
            and title.strip()
        ):
            out.setdefault(item_id.strip(), title.strip())
        for sub in value.values():
            _collect_known_titles(sub, out)
    elif isinstance(value, list | tuple):
        for elem in value:
            _collect_known_titles(elem, out)


class LibraryWorkflowNodes:
    """Encapsulates injected dependencies and handlers for the `library_agent` state machine."""

    def __init__(
        self,
        *,
        llm: ChatRunnable,
        gateway: LibraryGateway,
        tool_registry: LibraryToolRegistry,
        agent_config: ToolAgentConfig,
        event_publisher: ToolEventPublisher | None = None,
    ) -> None:
        self._llm = llm
        self._gateway = gateway
        self._tool_registry = tool_registry
        self._agent_config = agent_config
        self._events: ToolEventPublisher = event_publisher or LangGraphToolEvents("library_agent")
        self._catalog_text = tool_registry.format_catalog()
        self._planner_llm, self._native_tools_bound = bind_planner_tools(
            llm, tool_registry.to_openai_tools()
        )

    async def plan_and_act(self, state: LibraryState, config: RunnableConfig) -> dict[str, Any]:
        query = str(state.get("query") or "")
        next_step = int(state.get("step", 0)) + 1
        return await run_plan_step(
            self._planner_llm,
            system_prompt=build_planner_prompt(
                self._catalog_text,
                step=next_step,
                max_steps=self._agent_config.max_steps,
            ),
            user_content=format_conversation_tail(state.get("conversation_tail"))
            + format_observations_block(
                query=query,
                observations=state.get("observations") or [],
                attached_items=state.get("attached_items") or (),
            ),
            config=config,
            step=next_step,
            prior_usage=state.get("usage"),
            max_calls=4,
            native_tools_bound=self._native_tools_bound,
        )

    async def policy_gate(self, state: LibraryState, config: RunnableConfig) -> dict[str, Any]:
        del config
        observations = list(state.get("observations") or [])
        known_titles: dict[str, str] = {}
        _collect_known_titles(observations, known_titles)
        return run_policy_gate(
            pending=state.get("pending_calls") or [],
            observations=observations,
            get_spec=self._tool_registry.get,
            policy=self._agent_config.hitl_policy,
            step=int(state.get("step", 1)),
            id_prefix="lib",
            build_confirmation=lambda spec, iid, args: spec.build_confirmation(
                interrupt_id=iid, validated_args=args, known_titles=known_titles
            ),
        )

    async def execute_and_verify(
        self, state: LibraryState, config: RunnableConfig
    ) -> dict[str, Any]:
        scope = LibraryScope(
            space_id=str(state.get("space_id") or ""),
            user_id=str(state.get("user_id") or ""),
            scope_type=state.get("scope_type") or "auto",
        )
        outcome = await run_tool_batch(
            approved=state.get("approved_calls") or [],
            get_spec=self._tool_registry.get,
            invoke=lambda spec, args: spec.handler(self._gateway, scope, args),
            failure=lambda tool, msg: ToolResult(tool=tool, status="error", error=msg),
            gateway_error=LibraryGatewayError,
            events=self._events,
            config=config,
            timeout_seconds=self._agent_config.tool_timeout_seconds,
            observations=state.get("observations") or [],
            citations=state.get("citations") or [],
        )
        return {
            "approved_calls": [],
            "observations": outcome.observations,
            "citations": outcome.citations,
            "done": outcome.is_terminal,
        }

    async def synthesize_response(
        self, state: LibraryState, config: RunnableConfig
    ) -> dict[str, Any]:
        return await run_synthesize_step(
            self._llm,
            draft_answer=state.get("draft_answer"),
            system_prompt=build_synthesizer_prompt(),
            user_content=format_observations_block(
                query=str(state.get("query") or ""),
                observations=state.get("observations") or [],
                attached_items=state.get("attached_items") or (),
            ),
            conversation_tail=state.get("conversation_tail"),
            prior_usage=state.get("usage"),
            config=config,
        )


def build_library_nodes(
    *,
    llm: ChatRunnable,
    gateway: LibraryGateway,
    tool_registry: LibraryToolRegistry,
    agent_config: ToolAgentConfig,
    event_publisher: ToolEventPublisher | None = None,
) -> dict[str, Callable[..., Awaitable[dict[str, Any]]]]:
    """Build pure async LangGraph node callables backed by `LibraryWorkflowNodes`."""
    runner = LibraryWorkflowNodes(
        llm=llm,
        gateway=gateway,
        tool_registry=tool_registry,
        agent_config=agent_config,
        event_publisher=event_publisher,
    )
    return {
        "plan_and_act": runner.plan_and_act,
        "policy_gate": runner.policy_gate,
        "execute_and_verify": runner.execute_and_verify,
        "synthesize_response": runner.synthesize_response,
    }
