"""Application bootstrap and lifespan resource wiring."""

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from redis.asyncio import Redis

from ai.agents.editor.gateway.contracts import EditorGateway
from ai.agents.library.gateway.contracts import LibraryGateway
from ai.agents.project.gateway.contracts import ProjectGateway
from ai.agents.rag.ingestion.jobs import IngestionJobRunner, build_ingestion_job_store
from ai.agents.rag.ingestion.parsers import ParserRegistry
from ai.agents.rag.ingestion.pipeline import IngestionPipeline
from ai.agents.rag.retrieval.contracts import Reranker
from ai.agents.rag.retrieval.engine import RetrievalEngine
from ai.agents.web_search.fetch.ladder import UrlFetchLadder
from ai.agents.web_search.search.registry import SearchProviderRegistry
from ai.orchestrator.contracts import AgentRegistry, ContextWindowPolicy, InputGuard, OutputGuard
from ai.orchestrator.graph import build_root_graph
from ai.orchestrator.guards import CitationIntegrityGuard, DefaultInputGuard
from ai.runtime_builder import InfraBundle, build_agents, build_models, enter_infra
from ai.shared.config.server import AppEnvironment
from ai.shared.config.settings import Settings
from ai.shared.contracts.health import HealthProbe
from ai.shared.contracts.knowledge import Embedder, KnowledgeStore
from ai.shared.contracts.rate_limit import RateLimiter
from ai.shared.contracts.run_control import RunCancellationRegistry
from ai.shared.llm.factory import ChatRunnable
from ai.shared.persistence.health import build_infrastructure_probes
from ai.shared.persistence.rate_limiter import build_rate_limiter
from ai.shared.persistence.redis import await_redis
from ai.shared.persistence.run_control import build_run_cancellation
from ai.shared.resilience.circuit_breaker import CircuitBreakerRegistry
from ai.shared.resilience.mutations import JournalStore, build_journal
from ai.shared.resilience.observability import (
    LLMCallMetrics,
    TokenBudget,
    ToolExecutionMetrics,
    create_token_budget,
)

logger = logging.getLogger(__name__)
LifespanFactory = Callable[[FastAPI], AbstractAsyncContextManager[None]]


@dataclass(frozen=True, slots=True)
class AppContainer:
    """Immutable typed container holding all FastAPI lifespan runtime dependencies."""

    settings: Settings
    root_graph: CompiledStateGraph
    stateless_root_graph: CompiledStateGraph
    agent_registry: AgentRegistry
    checkpointer: BaseCheckpointSaver
    memory_store: BaseStore
    knowledge_store: KnowledgeStore
    ingestion_pipeline: IngestionPipeline
    ingestion_job_runner: IngestionJobRunner
    retrieval_engine: RetrievalEngine
    cancel_registry: RunCancellationRegistry
    rate_limiter: RateLimiter
    token_budget: TokenBudget
    llm_metrics: LLMCallMetrics
    tool_metrics: ToolExecutionMetrics
    circuit_breakers: CircuitBreakerRegistry
    mutation_journal: JournalStore
    shared_http_client: httpx.AsyncClient
    input_guards: tuple[InputGuard, ...]
    output_guards: tuple[OutputGuard, ...]
    health_probes: tuple[HealthProbe, ...]
    redis_client: Redis | None = None


@dataclass(frozen=True)
class RuntimeOverrides:
    """Optional dependency and configuration overrides for application startup."""

    llm: ChatRunnable | None = None
    router_llm: ChatRunnable | None = None
    registry: AgentRegistry | None = None
    context_policy: ContextWindowPolicy | None = None
    input_guards: Sequence[InputGuard] | None = None
    output_guards: Sequence[OutputGuard] | None = None
    max_rounds: int = 3
    rate_limiter: RateLimiter | None = None
    health_probes: Sequence[HealthProbe] | None = None
    embedder: Embedder | None = None
    knowledge_store: KnowledgeStore | None = None
    memory_store: BaseStore | None = None
    cancel_registry: RunCancellationRegistry | None = None
    reranker: Reranker | None = None
    parser_registry: ParserRegistry | None = None
    search_registry: SearchProviderRegistry | None = None
    fetch_ladder: UrlFetchLadder | None = None
    library_gateway: LibraryGateway | None = None
    editor_gateway: EditorGateway | None = None
    project_gateway: ProjectGateway | None = None


async def _run_bounded_warmup_step(
    name: str,
    awaitable: Awaitable[Any],
    *,
    timeout_seconds: float,
) -> None:
    """Run one startup warmup check within `timeout_seconds`, failing open on error."""
    try:
        await asyncio.wait_for(awaitable, timeout=timeout_seconds)
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


def build_lifespan(
    settings: Settings,
    *,
    overrides: RuntimeOverrides | None = None,
) -> LifespanFactory:
    """Build the FastAPI lifespan context manager for AI runtime resources."""
    ov = overrides or RuntimeOverrides()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        llm_metrics = LLMCallMetrics()
        models = build_models(
            settings,
            llm=ov.llm,
            router_llm=ov.router_llm,
            llm_metrics=llm_metrics,
        )
        input_guards: tuple[InputGuard, ...] = (
            tuple(ov.input_guards) if ov.input_guards is not None else (DefaultInputGuard(),)
        )
        output_guards: tuple[OutputGuard, ...] = (
            tuple(ov.output_guards) if ov.output_guards is not None else (CitationIntegrityGuard(),)
        )

        async with AsyncExitStack() as stack:
            infra = await enter_infra(
                stack,
                settings,
                memory_store=ov.memory_store,
                knowledge_store=ov.knowledge_store,
                embedder=ov.embedder,
            )
            agents = build_agents(
                settings,
                infra,
                models,
                parser_registry=ov.parser_registry,
                reranker=ov.reranker,
                search_registry=ov.search_registry,
                fetch_ladder=ov.fetch_ladder,
                library_gateway=ov.library_gateway,
                editor_gateway=ov.editor_gateway,
                project_gateway=ov.project_gateway,
            )
            active_registry = (
                ov.registry if ov.registry is not None else AgentRegistry(agents.specs)
            )
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

            cancel_registry = (
                ov.cancel_registry
                if ov.cancel_registry is not None
                else build_run_cancellation(infra.redis)
            )
            rate_limiter = (
                ov.rate_limiter
                if ov.rate_limiter is not None
                else build_rate_limiter(settings.rate_limit, infra.redis)
            )
            health_probes: tuple[HealthProbe, ...] = (
                tuple(ov.health_probes)
                if ov.health_probes is not None
                else build_infrastructure_probes(
                    settings.database,
                    infra.checkpointer,
                    infra.redis,
                    infra.qdrant,
                )
            )

            def _build_graph(active_cp: BaseCheckpointSaver | None) -> CompiledStateGraph:
                return build_root_graph(
                    llm=models.default,
                    router_llm=models.router,
                    registry=active_registry,
                    context_policy=ov.context_policy,
                    output_guards=output_guards,
                    max_rounds=ov.max_rounds,
                    checkpointer=active_cp,
                    store=infra.memory_store,
                )

            app.state.container = AppContainer(
                settings=settings,
                root_graph=_build_graph(infra.checkpointer),
                stateless_root_graph=_build_graph(None),
                agent_registry=active_registry,
                checkpointer=infra.checkpointer,
                memory_store=infra.memory_store,
                knowledge_store=infra.knowledge_store,
                ingestion_pipeline=agents.ingestion_pipeline,
                ingestion_job_runner=ingestion_job_runner,
                retrieval_engine=agents.retrieval_engine,
                cancel_registry=cancel_registry,
                rate_limiter=rate_limiter,
                token_budget=create_token_budget(infra.redis),
                llm_metrics=llm_metrics,
                tool_metrics=ToolExecutionMetrics(),
                circuit_breakers=agents.circuit_breakers,
                mutation_journal=build_journal(infra.redis),
                shared_http_client=infra.http,
                input_guards=input_guards,
                output_guards=output_guards,
                health_probes=health_probes,
                redis_client=infra.redis,
            )
            if settings.server.app_env != AppEnvironment.TEST:
                await _run_startup_warmup(infra)
            yield

    return lifespan
