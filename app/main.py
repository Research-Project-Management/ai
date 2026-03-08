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

    yield

    logger.info("👋 Flux-AI shutting down")


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
