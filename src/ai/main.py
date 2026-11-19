"""FastAPI application entrypoint."""

from collections.abc import Sequence

from fastapi import FastAPI
from langgraph.store.base import BaseStore

from ai.agents.editor.gateway.contracts import EditorGateway
from ai.agents.library.gateway.contracts import LibraryGateway
from ai.agents.project.gateway.contracts import ProjectGateway
from ai.agents.rag.ingestion.parsers import ParserRegistry
from ai.agents.rag.retrieval.contracts import Reranker
from ai.agents.web_search.fetch.ladder import UrlFetchLadder
from ai.agents.web_search.search.registry import SearchProviderRegistry
from ai.api.routers.agents import router as agents_router
from ai.api.routers.chat import router as chat_router
from ai.api.routers.feedback import router as feedback_router
from ai.api.routers.health import router as health_router
from ai.api.routers.knowledge import router as knowledge_router
from ai.api.routers.store import router as store_router
from ai.api.routers.threads import router as threads_router
from ai.bootstrap import build_lifespan, resolve_settings
from ai.orchestrator.contracts import AgentRegistry, ContextWindowPolicy, InputGuard, OutputGuard
from ai.shared.config.settings import Settings
from ai.shared.context.facade import ResearchContextFacade
from ai.shared.contracts.health import HealthProbe
from ai.shared.contracts.knowledge import Embedder, KnowledgeStore
from ai.shared.contracts.rate_limit import RateLimiter
from ai.shared.contracts.run_control import FeedbackSink, RunCancellationRegistry
from ai.shared.llm.factory import ChatRunnable


def create_app(
    settings: Settings | None = None,
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
) -> FastAPI:
    """Create the FastAPI application and attach routers."""
    active_settings = resolve_settings(settings)
    app = FastAPI(
        title="AI Multi-Agent Service",
        version="0.1.0",
        lifespan=build_lifespan(
            active_settings,
            llm=llm,
            router_llm=router_llm,
            registry=registry,
            context_policy=context_policy,
            input_guards=input_guards,
            output_guards=output_guards,
            max_rounds=max_rounds,
            rate_limiter=rate_limiter,
            health_probes=health_probes,
            embedder=embedder,
            knowledge_store=knowledge_store,
            memory_store=memory_store,
            cancel_registry=cancel_registry,
            feedback_sink=feedback_sink,
            reranker=reranker,
            parser_registry=parser_registry,
            search_registry=search_registry,
            fetch_ladder=fetch_ladder,
            library_gateway=library_gateway,
            editor_gateway=editor_gateway,
            project_gateway=project_gateway,
            context_facade=context_facade,
        ),
    )
    app.state.settings = active_settings
    for router in (
        health_router,
        chat_router,
        threads_router,
        store_router,
        knowledge_router,
        agents_router,
        feedback_router,
    ):
        app.include_router(router)
    return app


app = create_app()
