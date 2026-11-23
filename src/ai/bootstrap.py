"""Application bootstrap and lifespan resource wiring (Composition Root)."""

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from typing import Any

from fastapi import FastAPI
from langgraph.store.base import BaseStore

from ai.agents.editor.gateway.contracts import EditorGateway
from ai.agents.library.gateway.contracts import LibraryGateway
from ai.agents.project.gateway.contracts import ProjectGateway
from ai.agents.rag.ingestion.jobs import IngestionJobRunner, build_ingestion_job_store
from ai.agents.rag.ingestion.parsers import ParserRegistry
from ai.agents.rag.retrieval.contracts import Reranker
from ai.agents.web_search.fetch.ladder import UrlFetchLadder
from ai.agents.web_search.search.registry import SearchProviderRegistry
from ai.orchestrator.contracts import AgentRegistry, ContextWindowPolicy, InputGuard, OutputGuard
from ai.orchestrator.graph import build_root_graph
from ai.orchestrator.guards import CitationIntegrityGuard, DefaultInputGuard
from ai.runtime_builder import InfraBundle, build_agents, build_models, enter_infra
from ai.shared.config.server import AppEnvironment
from ai.shared.config.settings import Settings, load_settings
from ai.shared.context.facade import ResearchContextFacade
from ai.shared.contracts.health import HealthProbe
from ai.shared.contracts.knowledge import Embedder, KnowledgeStore
from ai.shared.contracts.rate_limit import RateLimiter
from ai.shared.contracts.run_control import FeedbackSink, RunCancellationRegistry
from ai.shared.llm.factory import ChatRunnable
from ai.shared.persistence.feedback import build_feedback_sink
from ai.shared.persistence.health import build_infrastructure_probes
from ai.shared.persistence.rate_limiter import build_rate_limiter
from ai.shared.persistence.redis import await_redis
from ai.shared.persistence.run_control import build_run_cancellation
from ai.shared.resilience.mutations import build_journal
from ai.shared.resilience.observability import (
    LLMCallMetrics,
    ToolExecutionMetrics,
    create_token_budget,
)

logger = logging.getLogger(__name__)
LifespanFactory = Callable[[FastAPI], AbstractAsyncContextManager[None]]


async def _run_bounded_warmup_step(
    name: str,
    awaitable: Awaitable[Any],
    *,
    timeout_seconds: float,
) -> None:
    """Run one startup warmup check within `timeout_seconds`, failing open on error."""

    async def _run() -> Any:
        return await awaitable

    try:
        await asyncio.wait_for(_run(), timeout=timeout_seconds)
    except Exception as exc:  # noqa: BLE001 - startup warmup is intentionally fail-open
        logger.warning("Startup warmup skipped for %s (%s)", name, exc)


async def _run_startup_warmup(
    infra: InfraBundle,
    *,
    timeout_seconds: float = 5.0,
) -> None:
    """Pre-warm Redis pool and Qdrant collection/alias concurrently without paid API calls."""
    steps: list[Coroutine[Any, Any, None]] = []
    if infra.redis is not None:
        steps.append(
            _run_bounded_warmup_step(
                "redis",
                await_redis(infra.redis.ping()),
                timeout_seconds=timeout_seconds,
            )
        )
    ensure_ready = getattr(infra.knowledge_store, "ensure_ready", None)
    if callable(ensure_ready):
        result = ensure_ready()
        if inspect.isawaitable(result):
            steps.append(
                _run_bounded_warmup_step(
                    "qdrant_knowledge",
                    result,
                    timeout_seconds=timeout_seconds,
                )
            )
    if steps:
        await asyncio.gather(*steps)


def resolve_settings(settings: Settings | None = None) -> Settings:
    """Return explicit settings if provided, otherwise load from environment."""
    return settings or load_settings()


def build_lifespan(
    settings: Settings,
    *,
    llm: ChatRunnable | None = None,
    router_llm: ChatRunnable | None = None,
    registry: AgentRegistry | None = None,
    context_policy: ContextWindowPolicy | None = None,
    input_guards: Sequence[InputGuard] | None = None,
    output_guards: Sequence[OutputGuard] | None = None,
    max_rounds: int = 3,
    rate_limiter: RateLimiter | None = None,
    health_probes: Sequence[HealthProbe] | None = None,
    embedder: Embedder | None = None,
    knowledge_store: KnowledgeStore | None = None,
    memory_store: BaseStore | None = None,
    cancel_registry: RunCancellationRegistry | None = None,
    feedback_sink: FeedbackSink | None = None,
    reranker: Reranker | None = None,
    parser_registry: ParserRegistry | None = None,
    search_registry: SearchProviderRegistry | None = None,
    fetch_ladder: UrlFetchLadder | None = None,
    library_gateway: LibraryGateway | None = None,
    editor_gateway: EditorGateway | None = None,
    project_gateway: ProjectGateway | None = None,
    context_facade: ResearchContextFacade | None = None,
) -> LifespanFactory:
    """Build the FastAPI lifespan context manager for AI runtime resources."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        llm_metrics = LLMCallMetrics()
        models = build_models(
            settings,
            llm=llm,
            router_llm=router_llm,
            llm_metrics=llm_metrics,
        )
        app.state.input_guards = (
            tuple(input_guards) if input_guards is not None else (DefaultInputGuard(),)
        )

        async with AsyncExitStack() as stack:
            infra = await enter_infra(
                stack,
                settings,
                memory_store=memory_store,
                knowledge_store=knowledge_store,
                embedder=embedder,
            )
            agents = build_agents(
                settings,
                infra,
                models,
                parser_registry=parser_registry,
                reranker=reranker,
                search_registry=search_registry,
                fetch_ladder=fetch_ladder,
                library_gateway=library_gateway,
                editor_gateway=editor_gateway,
                project_gateway=project_gateway,
                context_facade=context_facade,
            )
            active_registry = registry if registry is not None else AgentRegistry(agents.specs)

            app.state.checkpointer = infra.checkpointer
            app.state.memory_store = infra.memory_store
            app.state.knowledge_store = infra.knowledge_store
            app.state.ingestion_pipeline = agents.ingestion_pipeline
            ingestion_job_runner = IngestionJobRunner(
                agents.ingestion_pipeline,
                build_ingestion_job_store(
                    infra.redis,
                    ttl_seconds=settings.ingestion.job_ttl_seconds,
                ),
                max_concurrency=settings.ingestion.max_concurrency,
                max_queue_size=settings.ingestion.max_queue_size,
            )
            stack.push_async_callback(ingestion_job_runner.aclose)
            app.state.ingestion_job_runner = ingestion_job_runner
            app.state.retrieval_engine = agents.retrieval_engine
            app.state.agent_registry = active_registry
            app.state.cancel_registry = (
                cancel_registry
                if cancel_registry is not None
                else build_run_cancellation(infra.redis)
            )
            app.state.feedback_sink = (
                feedback_sink if feedback_sink is not None else build_feedback_sink(infra.redis)
            )
            app.state.rate_limiter = (
                rate_limiter
                if rate_limiter is not None
                else build_rate_limiter(settings.rate_limit, infra.redis)
            )
            app.state.circuit_breakers = agents.circuit_breakers
            app.state.shared_http_client = infra.http
            app.state.mutation_journal = build_journal(infra.redis)
            app.state.tool_metrics = ToolExecutionMetrics()
            app.state.llm_metrics = llm_metrics
            active_output_guards: Sequence[OutputGuard] = (
                tuple(output_guards) if output_guards is not None else (CitationIntegrityGuard(),)
            )
            app.state.output_guards = active_output_guards
            app.state.token_budget = create_token_budget(infra.redis)
            app.state.health_probes = (
                tuple(health_probes)
                if health_probes is not None
                else build_infrastructure_probes(
                    settings.database,
                    infra.checkpointer,
                    infra.redis,
                    infra.qdrant,
                )
            )
            app.state.context_facade = agents.context_facade

            def _build_graph(active_cp):
                return build_root_graph(
                    llm=models.default,
                    router_llm=models.router,
                    registry=active_registry,
                    context_policy=context_policy,
                    context_facade=agents.context_facade,
                    router_cache=agents.tool_cache,
                    output_guards=active_output_guards,
                    max_rounds=max_rounds,
                    checkpointer=active_cp,
                    store=infra.memory_store,
                )

            app.state.root_graph = _build_graph(infra.checkpointer)
            app.state.stateless_root_graph = _build_graph(None)
            if settings.server.app_env != AppEnvironment.TEST:
                await _run_startup_warmup(infra)
            yield

    return lifespan
