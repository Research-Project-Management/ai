"""Runtime builders for models, infrastructure clients, and specialist agents."""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass

import httpx
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from ai.agents.editor.gateway.contracts import EditorGateway
from ai.agents.editor.github.client import HttpGitHubArtifactClient
from ai.agents.editor.module import build_editor_module
from ai.agents.library.gateway.contracts import LibraryGateway
from ai.agents.library.module import build_library_module
from ai.agents.project.gateway.contracts import ProjectGateway
from ai.agents.project.module import build_project_module
from ai.agents.rag.ingestion.parsers import ParserRegistry
from ai.agents.rag.ingestion.pipeline import IngestionPipeline
from ai.agents.rag.module import build_rag_module
from ai.agents.rag.retrieval.contracts import Reranker
from ai.agents.rag.retrieval.engine import RetrievalEngine
from ai.agents.web_search.fetch.ladder import UrlFetchLadder
from ai.agents.web_search.module import build_web_search_module
from ai.agents.web_search.search.registry import SearchProviderRegistry
from ai.orchestrator.contracts import AgentSpec
from ai.shared.config.server import AppEnvironment
from ai.shared.config.settings import Settings
from ai.shared.contracts.knowledge import Embedder, KnowledgeStore
from ai.shared.llm.factory import ChatRunnable, ContextAwareChatRunnable, build_chat_model
from ai.shared.persistence.checkpointer import create_checkpointer
from ai.shared.persistence.knowledge import create_knowledge_store
from ai.shared.persistence.qdrant import create_qdrant_client
from ai.shared.persistence.redis import create_redis_client
from ai.shared.persistence.store import create_memory_store
from ai.shared.resilience.cache import create_tool_cache
from ai.shared.resilience.circuit_breaker import CircuitBreakerRegistry
from ai.shared.resilience.observability import LLMCallMetrics


@dataclass(frozen=True)
class ModelBundle:
    """Chat runnables resolved for default answers, routing, and specialist agents."""

    default: ChatRunnable
    router: ChatRunnable
    agent: ChatRunnable


@dataclass(frozen=True)
class InfraBundle:
    """Managed async IO clients and backing persistence stores for one lifespan."""

    http: httpx.AsyncClient
    checkpointer: BaseCheckpointSaver
    redis: Redis | None
    qdrant: AsyncQdrantClient | None
    memory_store: BaseStore
    knowledge_store: KnowledgeStore


@dataclass(frozen=True)
class AgentBundle:
    """Assembled specialist agent specs plus exposed RAG pipeline/engine services."""

    specs: list[AgentSpec]
    ingestion_pipeline: IngestionPipeline
    retrieval_engine: RetrievalEngine
    circuit_breakers: CircuitBreakerRegistry


def build_models(
    settings: Settings,
    *,
    llm: ChatRunnable | None = None,
    router_llm: ChatRunnable | None = None,
    llm_metrics: LLMCallMetrics | None = None,
) -> ModelBundle:
    """Construct default, router, and specialist chat runnables with test overrides."""
    default_llm: ChatRunnable = llm or ContextAwareChatRunnable(
        build_chat_model(settings.llm, role="default"),
        config=settings.llm,
        role="default",
        metrics=llm_metrics,
    )
    router_override = router_llm or llm
    active_router: ChatRunnable = (
        router_override
        if router_override is not None
        else ContextAwareChatRunnable(
            build_chat_model(settings.llm, role="router"),
            config=settings.llm,
            role="router",
            metrics=llm_metrics,
        )
    )
    agent_llm: ChatRunnable = llm or ContextAwareChatRunnable(
        build_chat_model(settings.llm, role="agent"),
        config=settings.llm,
        role="agent",
        metrics=llm_metrics,
    )
    return ModelBundle(default=default_llm, router=active_router, agent=agent_llm)


async def enter_infra(
    stack: AsyncExitStack,
    settings: Settings,
    *,
    memory_store: BaseStore | None = None,
    knowledge_store: KnowledgeStore | None = None,
    embedder: Embedder | None = None,
) -> InfraBundle:
    """Enter all async infrastructure context managers onto the shared exit stack."""
    http = await stack.enter_async_context(
        httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=settings.server.http_max_connections,
                max_keepalive_connections=settings.server.http_max_keepalive_connections,
                keepalive_expiry=settings.server.http_keepalive_expiry_seconds,
            ),
            timeout=settings.server.http_timeout_seconds,
        )
    )
    checkpointer = await stack.enter_async_context(create_checkpointer(settings.database.postgres))
    redis = await stack.enter_async_context(create_redis_client(settings.database.redis))
    qdrant = await stack.enter_async_context(create_qdrant_client(settings.database.qdrant))
    active_memory = (
        memory_store
        if memory_store is not None
        else await stack.enter_async_context(create_memory_store(settings.database.postgres))
    )
    active_knowledge = (
        knowledge_store
        if knowledge_store is not None
        else await stack.enter_async_context(
            create_knowledge_store(
                settings.embedding,
                settings.llm,
                qdrant,
                embedder=embedder,
                allow_remote_embeddings=(settings.server.app_env != AppEnvironment.TEST),
            )
        )
    )
    return InfraBundle(
        http=http,
        checkpointer=checkpointer,
        redis=redis,
        qdrant=qdrant,
        memory_store=active_memory,
        knowledge_store=active_knowledge,
    )


def build_agents(
    settings: Settings,
    infra: InfraBundle,
    models: ModelBundle,
    *,
    parser_registry: ParserRegistry | None = None,
    reranker: Reranker | None = None,
    search_registry: SearchProviderRegistry | None = None,
    fetch_ladder: UrlFetchLadder | None = None,
    library_gateway: LibraryGateway | None = None,
    editor_gateway: EditorGateway | None = None,
    project_gateway: ProjectGateway | None = None,
) -> AgentBundle:
    """Assemble all enabled specialist agent modules and shared services."""
    allow_remote = settings.server.app_env != AppEnvironment.TEST
    internal_key = settings.server.internal_api_key.get_secret_value()

    rag = build_rag_module(
        llm=models.agent,
        knowledge_store=infra.knowledge_store,
        ingestion_config=settings.ingestion,
        rerank_config=settings.rerank,
        parser_registry=parser_registry,
        reranker=reranker,
        allow_remote_rerank=allow_remote,
        backend_url=settings.library.backend_url,
        http_client=infra.http,
    )
    web_search = build_web_search_module(
        llm=models.agent,
        config=settings.web_search,
        redis_client=infra.redis,
        search_registry=search_registry,
        fetch_ladder=fetch_ladder,
        http_client=infra.http,
        allow_remote=allow_remote,
    )
    tool_cache = create_tool_cache(infra.redis)
    circuit_breakers = CircuitBreakerRegistry()
    github_client = HttpGitHubArtifactClient(
        http_client=infra.http,
        circuit_breaker=circuit_breakers.get_or_create("github"),
        cache=tool_cache,
    )

    specs: list[AgentSpec] = [rag.agent_spec, web_search.agent_spec]
    if settings.library.enabled:
        library = build_library_module(
            llm=models.agent,
            config=settings.library,
            gateway=library_gateway,
            http_client=infra.http,
        )
        specs.append(library.agent_spec)
    if settings.editor.enabled:
        editor = build_editor_module(
            llm=models.agent,
            config=settings.editor,
            gateway=editor_gateway,
            http_client=infra.http,
            internal_api_key=internal_key,
            artifact_client=github_client,
        )
        specs.append(editor.agent_spec)
    if settings.project.enabled:
        project = build_project_module(
            llm=models.agent,
            config=settings.project,
            gateway=project_gateway,
            http_client=infra.http,
            internal_api_key=internal_key,
        )
        specs.append(project.agent_spec)

    return AgentBundle(
        specs=specs,
        ingestion_pipeline=rag.ingestion_pipeline,
        retrieval_engine=rag.retrieval_engine,
        circuit_breakers=circuit_breakers,
    )
