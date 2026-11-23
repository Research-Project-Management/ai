"""SRP workflow node class implementing the 5-node `project_agent` state machine."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from ai.agents.project.gateway.contracts import (
    ProjectGateway,
    ProjectGatewayError,
    ProjectLabel,
    ProjectMember,
    ProjectScope,
    ProjectSummary,
    WorkflowState,
)
from ai.agents.project.resolution.context_builder import format_project_context
from ai.agents.project.resolution.contracts import ProjectContextSnapshot
from ai.agents.project.tools.contracts import (
    ProjectToolRegistry,
    ProjectToolResult,
    ProjectToolRuntime,
)
from ai.agents.project.workflow.contracts import (
    ProjectAgentConfig,
)
from ai.agents.project.workflow.prompts import (
    build_planner_prompt,
    build_synthesizer_prompt,
    format_observations_block,
)
from ai.agents.project.workflow.state import ProjectState
from ai.shared.contracts.run_control import RunUsage
from ai.shared.llm.factory import ChatRunnable
from ai.shared.llm.output import extract_text, strip_reasoning
from ai.shared.workflow.events import LangGraphToolEvents, ToolEventPublisher
from ai.shared.workflow.execute import run_tool_batch
from ai.shared.workflow.gate import build_confirmation_payload, run_policy_gate
from ai.shared.workflow.plan import run_plan_step

logger = logging.getLogger(__name__)


def _restore_models[T: BaseModel](raw_list: Any, cls: type[T]) -> list[T]:
    if not isinstance(raw_list, list):
        return []
    return [cls.model_validate(item) for item in raw_list if isinstance(item, dict)]


def _restore_model_map[T: BaseModel](raw_map: Any, cls: type[T]) -> dict[str, list[T]]:
    if not isinstance(raw_map, dict):
        return {}
    return {str(k): _restore_models(v, cls) for k, v in raw_map.items() if isinstance(v, list)}


class ProjectWorkflowNodes:
    """Encapsulates injected dependencies and node handlers for the `project_agent` subgraph."""

    def __init__(
        self,
        *,
        llm: ChatRunnable,
        gateway: ProjectGateway,
        tool_registry: ProjectToolRegistry,
        agent_config: ProjectAgentConfig,
        event_publisher: ToolEventPublisher | None = None,
    ) -> None:
        self._llm = llm
        self._gateway = gateway
        self._tool_registry = tool_registry
        self._agent_config = agent_config
        self._events: ToolEventPublisher = event_publisher or LangGraphToolEvents("project_agent")
        self._catalog_text = tool_registry.format_catalog()

    def _build_scope(self, state: ProjectState) -> ProjectScope:
        return ProjectScope(
            space_id=str(state.get("space_id") or ""),
            project_id=state.get("project_id"),
            user_id=str(state.get("user_id") or ""),
        )

    def _build_runtime(self, state: ProjectState) -> ProjectToolRuntime:
        return ProjectToolRuntime(
            gateway=self._gateway,
            scope=self._build_scope(state),
            projects=_restore_models(state.get("projects_cache"), ProjectSummary),
            states_by_project=_restore_model_map(state.get("states_cache"), WorkflowState),
            members_by_project=_restore_model_map(state.get("members_cache"), ProjectMember),
            labels_by_project=_restore_model_map(state.get("labels_cache"), ProjectLabel),
            widgets=list(state.get("widgets") or ()),
        )

    async def hydrate_context(self, state: ProjectState, config: RunnableConfig) -> dict[str, Any]:
        del config
        raw_ctx = dict(state.get("raw_context") or {})
        space_id = str(state.get("space_id") or raw_ctx.get("space_id") or "").strip()
        user_id = str(state.get("user_id") or raw_ctx.get("user_id") or "").strip()
        raw_pid = state.get("project_id") or raw_ctx.get("project_id")
        project_id = str(raw_pid).strip() if isinstance(raw_pid, str) and raw_pid.strip() else None

        scope = ProjectScope(space_id=space_id, project_id=project_id, user_id=user_id)
        projects: list[ProjectSummary] = _restore_models(
            state.get("projects_cache"), ProjectSummary
        )
        if not projects:
            try:
                projects = await self._gateway.projects.list_projects(scope)
            except (ProjectGatewayError, TimeoutError, OSError):
                projects = []

        if not project_id and len(projects) == 1:
            project_id = projects[0].id
            scope = ProjectScope(space_id=space_id, project_id=project_id, user_id=user_id)

        active_project: ProjectSummary | None = None
        states: list[WorkflowState] = []
        members: list[ProjectMember] = []
        labels: list[ProjectLabel] = []

        if project_id:
            active_project = next(
                (p for p in projects if p.id == project_id or p.identifier == project_id),
                None,
            )
            try:
                if active_project is None:
                    active_project = await self._gateway.projects.get_project(scope, project_id)
                project_id = active_project.id or project_id
                states = await self._gateway.projects.list_workflow_states(scope, project_id)
                members = await self._gateway.projects.list_members(scope, project_id, limit=100)
                labels = await self._gateway.projects.list_labels(scope, project_id)
            except (ProjectGatewayError, TimeoutError, OSError):
                pass

        snapshot = ProjectContextSnapshot(
            active_project=active_project,
            projects=tuple(projects),
            workflow_states=tuple(states),
            members=tuple(members),
            labels=tuple(labels),
            user_id=user_id,
            space_id=space_id,
        )
        snapshot_text = format_project_context(snapshot)

        states_cache = dict(state.get("states_cache") or {})
        members_cache = dict(state.get("members_cache") or {})
        labels_cache = dict(state.get("labels_cache") or {})
        if project_id:
            if states:
                states_cache[project_id] = [s.model_dump(mode="json") for s in states]
            if members:
                members_cache[project_id] = [m.model_dump(mode="json") for m in members]
            if labels:
                labels_cache[project_id] = [lb.model_dump(mode="json") for lb in labels]

        return {
            "project_id": project_id,
            "context_snapshot_text": snapshot_text,
            "projects_cache": [p.model_dump(mode="json") for p in projects],
            "states_cache": states_cache,
            "members_cache": members_cache,
            "labels_cache": labels_cache,
        }

    async def plan_and_act(self, state: ProjectState, config: RunnableConfig) -> dict[str, Any]:
        query = str(state.get("query") or "")
        tail = state.get("conversation_tail") or []
        tail_block = "Recent conversation:\n" + "\n".join(tail) + "\n\n" if tail else ""
        next_step = int(state.get("step", 0)) + 1
        return await run_plan_step(
            self._llm,
            system_prompt=build_planner_prompt(
                self._catalog_text,
                step=next_step,
                max_steps=self._agent_config.max_steps,
            ),
            user_content=tail_block
            + format_observations_block(
                query=query,
                context_snapshot_text=str(state.get("context_snapshot_text") or ""),
                observations=state.get("observations") or (),
                peer_findings=state.get("peer_findings") or (),
            ),
            config=config,
            step=next_step,
            prior_usage=state.get("usage"),
            max_calls=6,
        )

    async def policy_gate(self, state: ProjectState, config: RunnableConfig) -> dict[str, Any]:
        del config
        return run_policy_gate(
            pending=state.get("pending_calls") or [],
            observations=state.get("observations") or [],
            get_spec=self._tool_registry.get,
            policy=self._agent_config.hitl_policy,
            step=int(state.get("step", 1)),
            id_prefix="pj",
            build_confirmation=lambda spec, iid, args: build_confirmation_payload(
                interrupt_id=iid, spec=spec, validated_args=args
            ),
        )

    async def execute_and_verify(
        self, state: ProjectState, config: RunnableConfig
    ) -> dict[str, Any]:
        runtime = self._build_runtime(state)
        outcome = await run_tool_batch(
            approved=state.get("approved_calls") or [],
            get_spec=self._tool_registry.get,
            invoke=lambda spec, args: spec.handler(runtime, args),
            failure=lambda tool, msg: ProjectToolResult(tool=tool, status="error", error=msg),
            gateway_error=ProjectGatewayError,
            events=self._events,
            config=config,
            timeout_seconds=self._agent_config.tool_timeout_seconds,
            observations=state.get("observations") or [],
            citations=state.get("citations") or [],
        )
        return {
            "approved_calls": [],
            "observations": outcome.observations,
            "projects_cache": [p.model_dump(mode="json") for p in runtime.projects],
            "states_cache": {
                k: [s.model_dump(mode="json") for s in v]
                for k, v in runtime.states_by_project.items()
            },
            "members_cache": {
                k: [m.model_dump(mode="json") for m in v]
                for k, v in runtime.members_by_project.items()
            },
            "labels_cache": {
                k: [lb.model_dump(mode="json") for lb in v]
                for k, v in runtime.labels_by_project.items()
            },
            "widgets": runtime.widgets,
            "citations": outcome.citations,
            "done": outcome.is_terminal,
        }

    async def synthesize_response(
        self, state: ProjectState, config: RunnableConfig
    ) -> dict[str, Any]:
        draft = str(state.get("draft_answer") or "").strip()
        widgets = state.get("widgets") or []
        peer_findings = state.get("peer_findings") or []
        observations = state.get("observations") or []

        if draft and not widgets:
            return {"final_answer": draft}

        query = str(state.get("query") or "")
        tail = state.get("conversation_tail") or []
        tail_block = "Recent conversation:\n" + "\n".join(tail) + "\n\n" if tail else ""
        user_content = tail_block + format_observations_block(
            query=query,
            context_snapshot_text=str(state.get("context_snapshot_text") or ""),
            observations=observations,
            peer_findings=peer_findings,
        )

        response = await self._llm.ainvoke(
            [
                SystemMessage(content=build_synthesizer_prompt(has_widgets=bool(widgets))),
                HumanMessage(content=user_content),
            ],
            config=config,
        )
        prior_usage = state.get("usage") or RunUsage()
        return {
            "final_answer": strip_reasoning(extract_text(response.content)),
            "usage": prior_usage.merge(RunUsage.from_message(response)),
        }


def build_project_nodes(
    *,
    llm: ChatRunnable,
    gateway: ProjectGateway,
    tool_registry: ProjectToolRegistry,
    agent_config: ProjectAgentConfig,
    event_publisher: ToolEventPublisher | None = None,
) -> dict[str, Callable[..., Awaitable[dict[str, Any]]]]:
    """Build pure async LangGraph node callables for `project_agent`."""
    runner = ProjectWorkflowNodes(
        llm=llm,
        gateway=gateway,
        tool_registry=tool_registry,
        agent_config=agent_config,
        event_publisher=event_publisher,
    )
    return {
        "hydrate_context": runner.hydrate_context,
        "plan_and_act": runner.plan_and_act,
        "policy_gate": runner.policy_gate,
        "execute_and_verify": runner.execute_and_verify,
        "synthesize_response": runner.synthesize_response,
    }
