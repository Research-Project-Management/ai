"""SRP workflow node class implementing the 5-node `editor_agent` state machine."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast

from langchain_core.runnables import RunnableConfig

from ai.agents.editor.context.builder import (
    build_editor_context_snapshot,
    format_context_for_prompt,
)
from ai.agents.editor.context.latex_scanner import normalize_newlines
from ai.agents.editor.gateway.contracts import (
    EditorGateway,
    EditorGatewayError,
    EditorScope,
)
from ai.agents.editor.patching.contracts import EditorPatchProposal
from ai.agents.editor.tools.contracts import (
    CachedFile,
    EditorToolRegistry,
    EditorToolResult,
    EditorToolRuntime,
)
from ai.agents.editor.workflow.contracts import EditorAgentConfig
from ai.agents.editor.workflow.events import (
    LangGraphEditorEvents,
    NullEditorEvents,
)
from ai.agents.editor.workflow.prompts import (
    build_planner_prompt,
    build_synthesizer_prompt,
    format_observations_block,
)
from ai.agents.editor.workflow.state import EditorState
from ai.shared.llm.factory import ChatRunnable
from ai.shared.workflow.execute import run_tool_batch
from ai.shared.workflow.gate import build_confirmation_payload, run_policy_gate
from ai.shared.workflow.plan import run_plan_step
from ai.shared.workflow.synthesize import format_conversation_tail, run_synthesize_step

logger = logging.getLogger(__name__)


def _restore_file_cache(raw_cache: dict[str, dict[str, Any]] | None) -> dict[str, CachedFile]:
    restored: dict[str, CachedFile] = {}
    for key, item in (raw_cache or {}).items():
        if isinstance(item, dict):
            restored[key] = CachedFile.model_validate(item)
    return restored


def _serialize_file_cache(cache: dict[str, CachedFile]) -> dict[str, dict[str, Any]]:
    return {key: entry.model_dump() for key, entry in cache.items()}


def _restore_patches(raw_patches: list[dict[str, Any]] | None) -> list[EditorPatchProposal]:
    restored: list[EditorPatchProposal] = []
    for item in raw_patches or []:
        if isinstance(item, dict):
            restored.append(EditorPatchProposal.model_validate(item))
    return restored


def _serialize_patches(patches: list[EditorPatchProposal]) -> list[dict[str, Any]]:
    return [p.model_dump(mode="json") for p in patches]


class EditorWorkflowNodes:
    """Encapsulates injected dependencies and node handlers for the `editor_agent` subgraph."""

    def __init__(
        self,
        *,
        llm: ChatRunnable,
        gateway: EditorGateway,
        tool_registry: EditorToolRegistry,
        agent_config: EditorAgentConfig,
        event_publisher: LangGraphEditorEvents | NullEditorEvents | None = None,
    ) -> None:
        self._llm = llm
        self._gateway = gateway
        self._tool_registry = tool_registry
        self._agent_config = agent_config
        self._events: LangGraphEditorEvents | NullEditorEvents = (
            event_publisher or LangGraphEditorEvents()
        )
        self._catalog_text = tool_registry.format_catalog()
        self._openai_tools = tool_registry.to_openai_tools()
        self._planner_llm: ChatRunnable = llm
        self._native_tools_bound = False
        bind_fn = getattr(llm, "bind_tools", None)
        if self._openai_tools and callable(bind_fn):
            try:
                self._planner_llm = cast(ChatRunnable, bind_fn(self._openai_tools))
                self._native_tools_bound = True
            except (NotImplementedError, TypeError):
                self._planner_llm = llm
                self._native_tools_bound = False

    def _build_scope(
        self,
        state: EditorState,
        *,
        active_file_id: str | None = None,
        active_file_path: str | None = None,
    ) -> EditorScope:
        return EditorScope(
            page_id=str(state.get("page_id") or ""),
            space_id=str(state.get("space_id") or ""),
            user_id=str(state.get("user_id") or ""),
            active_file_id=(
                active_file_id if active_file_id is not None else state.get("active_file_id")
            ),
            active_file_path=active_file_path or state.get("active_file_path") or "main.tex",
        )

    def _build_runtime(self, state: EditorState) -> EditorToolRuntime:
        return EditorToolRuntime(
            gateway=self._gateway,
            scope=self._build_scope(state),
            file_cache=_restore_file_cache(state.get("file_cache")),
            known_cite_keys=set(state.get("known_cite_keys") or ()),
            known_labels=set(state.get("known_labels") or ()),
            patches=_restore_patches(state.get("patches")),
            max_patches=self._agent_config.max_patches_per_turn,
        )

    async def _fetch_fallback_active_file(
        self,
        scope: EditorScope,
        *,
        active_file_id: str | None,
        active_file_path: str,
    ) -> tuple[str | None, str, str | None, bool]:
        try:
            target_id = active_file_id
            resolved_path = active_file_path
            if not target_id:
                files = await self._gateway.list_files(scope, include_content=False)
                exact = [
                    f
                    for f in files
                    if f.path.lstrip("/") == active_file_path or f.title == active_file_path
                ]
                if exact:
                    target_id = exact[0].id
                    resolved_path = exact[0].path.lstrip("/") or active_file_path
                elif files:
                    root = next((f for f in files if f.is_root), files[0])
                    target_id = root.id
                    resolved_path = root.path.lstrip("/") or active_file_path
            if target_id:
                doc = await self._gateway.read_file(scope, target_id)
                return (
                    doc.id,
                    doc.path.lstrip("/") or resolved_path,
                    doc.content,
                    True,
                )
        except (EditorGatewayError, TimeoutError, OSError):
            # Graceful fallback when Backend token/page is unavailable (e.g. stateless preview)
            pass
        return active_file_id, active_file_path, None, False

    async def hydrate_context(self, state: EditorState, config: RunnableConfig) -> dict[str, Any]:
        del config
        raw_ctx = dict(state.get("raw_editor_context") or {})
        page_id = str(state.get("page_id") or raw_ctx.get("page_id") or "").strip()
        space_id = str(state.get("space_id") or raw_ctx.get("space_id") or "").strip()
        user_id = str(state.get("user_id") or raw_ctx.get("user_id") or "").strip()

        active_file_id = str(raw_ctx.get("active_file_id") or "").strip() or None
        active_file_path = (
            str(raw_ctx.get("active_file_path") or raw_ctx.get("filename") or "")
            .strip()
            .lstrip("/")
            or "main.tex"
        )

        raw_file_content = raw_ctx.get("file_content")
        has_live_content = isinstance(raw_file_content, str) and len(raw_file_content) > 0
        hydrated_override: str | None = None
        fetched_from_backend = False

        if not has_live_content and page_id:
            scope = EditorScope(
                page_id=page_id,
                space_id=space_id,
                user_id=user_id,
                active_file_id=active_file_id,
                active_file_path=active_file_path,
            )
            (
                active_file_id,
                active_file_path,
                hydrated_override,
                fetched_from_backend,
            ) = await self._fetch_fallback_active_file(
                scope,
                active_file_id=active_file_id,
                active_file_path=active_file_path,
            )

        merged_ctx = {
            **raw_ctx,
            **({"page_id": page_id} if page_id else {}),
            **({"active_file_id": active_file_id} if active_file_id else {}),
            "active_file_path": active_file_path,
        }
        snapshot = build_editor_context_snapshot(
            merged_ctx,
            active_content_override=hydrated_override,
            viewport_radius_lines=self._agent_config.viewport_radius_lines,
            max_chars=min(self._agent_config.max_file_chars, 16_000),
        )
        snapshot_text = format_context_for_prompt(snapshot)

        file_cache: dict[str, dict[str, Any]] = dict(state.get("file_cache") or {})
        effective_content = (
            hydrated_override
            if fetched_from_backend and hydrated_override is not None
            else (normalize_newlines(raw_file_content) if isinstance(raw_file_content, str) else "")
        )
        if has_live_content or fetched_from_backend:
            file_cache[snapshot.active_file_path] = CachedFile(
                file_id=snapshot.active_file_id,
                file_path=snapshot.active_file_path,
                content=effective_content,
            ).model_dump()

        return {
            "page_id": page_id,
            "active_file_id": snapshot.active_file_id,
            "active_file_path": snapshot.active_file_path,
            "context_snapshot_text": snapshot_text,
            "file_cache": file_cache,
            "known_cite_keys": list(state.get("known_cite_keys") or ()),
            "known_labels": list(state.get("known_labels") or ()),
        }

    async def plan_and_act(self, state: EditorState, config: RunnableConfig) -> dict[str, Any]:
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
                context_snapshot_text=str(state.get("context_snapshot_text") or ""),
                observations=state.get("observations") or (),
                patches=state.get("patches") or (),
                peer_findings=state.get("peer_findings") or (),
            ),
            config=config,
            step=next_step,
            prior_usage=state.get("usage"),
            max_calls=6,
            native_tools_bound=self._native_tools_bound,
        )

    async def policy_gate(self, state: EditorState, config: RunnableConfig) -> dict[str, Any]:
        del config
        return run_policy_gate(
            pending=state.get("pending_calls") or [],
            observations=state.get("observations") or [],
            get_spec=self._tool_registry.get,
            policy=self._agent_config.hitl_policy,
            step=int(state.get("step", 1)),
            id_prefix="ed",
            build_confirmation=lambda spec, iid, args: build_confirmation_payload(
                interrupt_id=iid, spec=spec, validated_args=args
            ),
        )

    async def execute_and_verify(
        self, state: EditorState, config: RunnableConfig
    ) -> dict[str, Any]:
        runtime = self._build_runtime(state)

        async def preview_patch(result: EditorToolResult) -> None:
            if result.status == "ok" and result.patch is not None:
                await self._events.emit_patch_preview(result.patch, len(runtime.patches), config)

        outcome = await run_tool_batch(
            approved=state.get("approved_calls") or [],
            get_spec=self._tool_registry.get,
            invoke=lambda spec, args: spec.handler(runtime, args),
            failure=lambda tool, msg: EditorToolResult(tool=tool, status="error", error=msg),
            gateway_error=EditorGatewayError,
            events=self._events,
            config=config,
            timeout_seconds=self._agent_config.tool_timeout_seconds,
            observations=state.get("observations") or [],
            citations=state.get("citations") or [],
            after_result=preview_patch,
        )
        return {
            "approved_calls": [],
            "observations": outcome.observations,
            "file_cache": _serialize_file_cache(runtime.file_cache),
            "known_cite_keys": sorted(runtime.known_cite_keys),
            "known_labels": sorted(runtime.known_labels),
            "patches": _serialize_patches(runtime.patches),
            "citations": outcome.citations,
            "done": outcome.is_terminal,
        }

    async def synthesize_response(
        self, state: EditorState, config: RunnableConfig
    ) -> dict[str, Any]:
        patches = state.get("patches") or []
        return await run_synthesize_step(
            self._llm,
            draft_answer=state.get("draft_answer"),
            allow_draft_fast_path=not patches,
            system_prompt=build_synthesizer_prompt(
                has_patches=bool(patches),
                has_citations=bool(state.get("citations")),
            ),
            user_content=format_observations_block(
                query=str(state.get("query") or ""),
                context_snapshot_text=str(state.get("context_snapshot_text") or ""),
                observations=state.get("observations") or (),
                patches=patches,
                peer_findings=state.get("peer_findings") or (),
            ),
            conversation_tail=state.get("conversation_tail"),
            prior_usage=state.get("usage"),
            config=config,
        )


def build_editor_nodes(
    *,
    llm: ChatRunnable,
    gateway: EditorGateway,
    tool_registry: EditorToolRegistry,
    agent_config: EditorAgentConfig,
    event_publisher: LangGraphEditorEvents | NullEditorEvents | None = None,
) -> dict[str, Callable[..., Awaitable[dict[str, Any]]]]:
    """Build pure async LangGraph node callables for `editor_agent`."""
    runner = EditorWorkflowNodes(
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
