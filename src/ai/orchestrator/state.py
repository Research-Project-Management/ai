"""Shared RootState flowing across the orchestrator and specialist subgraphs."""

from typing import Annotated, Any, Final

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from ai.orchestrator.contracts import AgentOutput, CitationItem, RoutingDecision
from ai.shared.contracts.run_control import RunUsage

RESET_AGENT_OUTPUTS: Final[str] = "__RESET__"
RESET_RUN_USAGE: Final[str] = "__RESET_USAGE__"


def reduce_agent_outputs(
    existing: list[AgentOutput] | None,
    update: list[AgentOutput] | str,
) -> list[AgentOutput]:
    """Accumulate parallel/multi-round specialist outputs within a turn, or reset between turns."""
    if update == RESET_AGENT_OUTPUTS:
        return []
    if isinstance(update, list):
        return [*(existing or []), *update]
    return list(existing or [])


def reduce_run_usage(
    existing: RunUsage | None,
    update: RunUsage | str,
) -> RunUsage:
    """Accumulate token counts within a turn, or reset between turns."""
    if update == RESET_RUN_USAGE:
        return RunUsage()
    if isinstance(update, RunUsage):
        base = existing or RunUsage()
        return base.merge(update)
    return existing or RunUsage()


class RootState(TypedDict, total=False):
    """Shared state for the root orchestration graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    intent_hint: str | None
    context: dict[str, Any]
    round_count: int
    plan_summary: str
    executed_rounds: list[str]
    routing_decision: RoutingDecision | None
    route: str
    active_agent: str
    active_instruction: str
    agent_outputs: Annotated[list[AgentOutput], reduce_agent_outputs]
    citations: list[CitationItem]
    artifacts: dict[str, Any]
    usage: Annotated[RunUsage, reduce_run_usage]
    final_response: str
