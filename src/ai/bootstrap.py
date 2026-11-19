"""Application bootstrap and lifespan resource wiring (Composition Root)."""

from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager

import httpx
from fastapi import FastAPI
from langgraph.store.base import BaseStore

from ai.agents.editor.gateway.contracts import EditorGateway
from ai.agents.editor.module import build_editor_module
from ai.agents.library.gateway.contracts import LibraryGateway
from ai.agents.library.module import build_library_module
from ai.agents.project.gateway.contracts import ProjectGateway
from ai.agents.project.module import build_project_module
from ai.agents.rag.ingestion.parsers import ParserRegistry
from ai.agents.rag.module import build_rag_module
from ai.agents.rag.retrieval.contracts import Reranker
from ai.agents.web_search.fetch.ladder import UrlFetchLadder
from ai.agents.web_search.module import build_web_search_module
from ai.agents.web_search.search.registry import SearchProviderRegistry
from ai.orchestrator.contracts import AgentRegistry, ContextWindowPolicy, InputGuard, OutputGuard
from ai.orchestrator.graph import build_root_graph
from ai.orchestrator.guards import CitationIntegrityGuard, DefaultInputGuard
from ai.shared.config.server import AppEnvironment
from ai.shared.config.settings import Settings, load_settings
from ai.shared.context.facade import ResearchContextFacade
from ai.shared.contracts.health import HealthProbe
from ai.shared.contracts.knowledge import Embedder, KnowledgeStore
from ai.shared.contracts.rate_limit import RateLimiter
from ai.shared.contracts.run_control import FeedbackSink, RunCancellationRegistry
from ai.shared.github.client import HttpGitHubArtifactClient
from ai.shared.llm.factory import ChatRunnable, ContextAwareChatRunnable, build_chat_model
from ai.shared.persistence.checkpointer import create_checkpointer
from ai.shared.persistence.feedback import build_feedback_sink
from ai.shared.persistence.health import build_infrastructure_probes
from ai.shared.persistence.knowledge import create_knowledge_store
from ai.shared.persistence.qdrant import create_qdrant_client
from ai.shared.persistence.rate_limiter import build_rate_limiter
from ai.shared.persistence.redis import create_redis_client
from ai.shared.persistence.run_control import build_run_cancellation
from ai.shared.persistence.store import create_memory_store
from ai.shared.resilience.cache import create_tool_cache
from ai.shared.resilience.circuit_breaker import CircuitBreakerRegistry
from ai.shared.resilience.observability import ToolExecutionMetrics, create_token_budget

LifespanFactory = Callable[[FastAPI], AbstractAsyncContextManager[None]]


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
        active_llm: ChatRunnable = llm or ContextAwareChatRunnable(
            build_chat_model(settings.llm, role="default"),
            config=settings.llm,
            role="default",
        )
        active_router_llm: ChatRunnable = router_llm or (
            llm if llm is not None else build_chat_model(settings.llm, role="router")
        )
        active_agent_llm: ChatRunnable = llm or ContextAwareChatRunnable(
            build_chat_model(settings.llm, role="agent"),
            config=settings.llm,
            role="agent",
        )
        active_input_guards = (
            tuple(input_guards) if input_guards is not None else (DefaultInputGuard(),)
        )
        app.state.input_guards = active_input_guards

        async with AsyncExitStack() as stack:
            shared_http_client = await stack.enter_async_context(
                httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=200,
                        max_keepalive_connections=50,
                        keepalive_expiry=30.0,
                    ),
                    timeout=15.0,
                )
            )
            checkpointer = await stack.enter_async_context(
                create_checkpointer(settings.database.postgres)
            )
            redis_client = await stack.enter_async_context(
                create_redis_client(settings.database.redis)
            )
            qdrant_client = await stack.enter_async_context(
                create_qdrant_client(settings.database.qdrant)
            )
            active_memory_store = (
                memory_store
                if memory_store is not None
                else await stack.enter_async_context(
                    create_memory_store(settings.database.postgres)
                )
            )
            active_knowledge_store = (
                knowledge_store
                if knowledge_store is not None
                else await stack.enter_async_context(
                    create_knowledge_store(
                        settings.embedding,
                        settings.llm,
                        qdrant_client,
                        embedder=embedder,
                        allow_remote_embeddings=(settings.server.app_env != AppEnvironment.TEST),
                    )
                )
            )

            rag_module = build_rag_module(
                llm=active_agent_llm,
                knowledge_store=active_knowledge_store,
                ingestion_config=settings.ingestion,
                rerank_config=settings.rerank,
                parser_registry=parser_registry,
                reranker=reranker,
                allow_remote_rerank=(settings.server.app_env != AppEnvironment.TEST),
                backend_url=settings.library.backend_url,
                http_client=shared_http_client,
            )
            web_search_module = build_web_search_module(
                llm=active_agent_llm,
                config=settings.web_search,
                redis_client=redis_client,
                search_registry=search_registry,
                fetch_ladder=fetch_ladder,
                http_client=shared_http_client,
                allow_remote=(settings.server.app_env != AppEnvironment.TEST),
            )
            library_module = build_library_module(
                llm=active_agent_llm,
                config=settings.library,
                gateway=library_gateway,
                http_client=shared_http_client,
            )
            tool_cache = create_tool_cache(redis_client)
            circuit_breakers = CircuitBreakerRegistry()
            active_context_facade = context_facade or ResearchContextFacade()
            github_artifact_client = HttpGitHubArtifactClient(
                http_client=shared_http_client,
                circuit_breaker=circuit_breakers.get_or_create("github"),
                cache=tool_cache,
            )
            editor_module = build_editor_module(
                llm=active_agent_llm,
                config=settings.editor,
                gateway=editor_gateway,
                http_client=shared_http_client,
                internal_api_key=settings.server.internal_api_key.get_secret_value(),
                artifact_client=github_artifact_client,
                context_facade=active_context_facade,
            )
            project_module = build_project_module(
                llm=active_agent_llm,
                config=settings.project,
                gateway=project_gateway,
                http_client=shared_http_client,
                internal_api_key=settings.server.internal_api_key.get_secret_value(),
            )
            default_specs = [
                rag_module.agent_spec,
                web_search_module.agent_spec,
                *(
                    (library_module.agent_spec,)
                    if settings.library.enabled
                    else ()
                ),
                *(
                    (editor_module.agent_spec,)
                    if settings.editor.enabled
                    else ()
                ),
                *(
                    (project_module.agent_spec,)
                    if settings.project.enabled
                    else ()
                ),
            ]
            active_registry = (
                registry if registry is not None else AgentRegistry(default_specs)
            )

            app.state.checkpointer = checkpointer
            app.state.memory_store = active_memory_store
            app.state.knowledge_store = active_knowledge_store
            app.state.ingestion_pipeline = rag_module.ingestion_pipeline
            app.state.retrieval_engine = rag_module.retrieval_engine
            app.state.agent_registry = active_registry
            app.state.cancel_registry = (
                cancel_registry
                if cancel_registry is not None
                else build_run_cancellation(redis_client)
            )
            app.state.feedback_sink = (
                feedback_sink
                if feedback_sink is not None
                else build_feedback_sink(redis_client)
            )
            app.state.rate_limiter = (
                rate_limiter
                if rate_limiter is not None
                else build_rate_limiter(settings.rate_limit, redis_client)
            )
            app.state.circuit_breakers = circuit_breakers
            app.state.tool_metrics = ToolExecutionMetrics()
            active_output_guards: Sequence[OutputGuard] = (
                tuple(output_guards)
                if output_guards is not None
                else (CitationIntegrityGuard(),)
            )
            app.state.output_guards = active_output_guards
            token_budget = create_token_budget(redis_client)
            app.state.token_budget = token_budget
            app.state.health_probes = (
                tuple(health_probes)
                if health_probes is not None
                else build_infrastructure_probes(
                    settings.database,
                    checkpointer,
                    redis_client,
                    qdrant_client,
                )
            )
            app.state.context_facade = active_context_facade
            app.state.root_graph = build_root_graph(
                llm=active_llm,
                router_llm=active_router_llm,
                registry=active_registry,
                context_policy=context_policy,
                context_facade=active_context_facade,
                router_cache=tool_cache,
                output_guards=active_output_guards,
                max_rounds=max_rounds,
                checkpointer=checkpointer,
                store=active_memory_store,
            )
            app.state.stateless_root_graph = build_root_graph(
                llm=active_llm,
                router_llm=active_router_llm,
                registry=active_registry,
                context_policy=context_policy,
                context_facade=active_context_facade,
                router_cache=tool_cache,
                output_guards=active_output_guards,
                max_rounds=max_rounds,
                checkpointer=None,
                store=active_memory_store,
            )
            yield

    return lifespan
