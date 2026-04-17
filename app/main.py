"""
Flux-AI — FastAPI entry point.

Multi-agent orchestrator for scientific research task management.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown events."""
    import asyncio
    settings = get_settings()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if settings.debug else logging.INFO,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    )

    logger.info("🚀 Flux-AI starting on %s:%s", settings.host, settings.port)

    # Import agents to trigger registration
    import app.agents  # noqa: F401

    from app.agents.registry import list_agents

    registered = list_agents()
    logger.info(
        "📦 Registered agents: %s",
        ", ".join(registered.keys()),
    )

    # Pre-warm CrossEncoder reranker in background (non-blocking startup).
    # Without this, the first RAG request would block while downloading ~2.27GB.
    if settings.enable_reranking:
        async def _prewarm_reranker():
            try:
                await asyncio.to_thread(__import__("app.agents.graph", fromlist=["_get_reranker"])._get_reranker)
                logger.info("🔥 CrossEncoder reranker pre-warmed.")
            except Exception as exc:
                logger.warning("⚠️  CrossEncoder pre-warm failed (non-fatal): %s", exc)

        asyncio.create_task(_prewarm_reranker())

    # Pre-warm orchestrator imports (sub-agents, tools) to avoid cold-start latency
    try:
        import app.agents.orchestrator  # noqa: F401
        logger.info("🔥 Orchestrator pre-imported.")
    except Exception as exc:
        logger.warning("Orchestrator pre-import failed: %s", exc)

    yield

    # ── Shutdown ──────────────────────────────────────────────────
    logger.info("👋 Flux-AI shutting down")

    # Gracefully close the httpx connection pool
    try:
        from app.tools._client import flux_api
        await flux_api.close()
        logger.info("🔌 HTTP connection pool closed.")
    except Exception:
        pass


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Flux-AI",
        description="Multi-Agent AI Orchestrator for Scientific Research",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routes
    from app.api.chat import router as chat_router
    from app.api.documents import router as documents_router
    from app.api.health import router as health_router

    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(documents_router)

    @app.get("/")
    async def root():
        return {
            "name": "Flux-AI",
            "version": "0.1.0",
            "docs": "/docs",
        }

    return app


# Module-level app for uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
