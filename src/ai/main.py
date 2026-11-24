"""FastAPI application entrypoint."""

from typing import Any

from fastapi import FastAPI

from ai.api.routers.agents import router as agents_router
from ai.api.routers.chat import router as chat_router
from ai.api.routers.health import router as health_router
from ai.api.routers.knowledge import router as knowledge_router
from ai.api.routers.threads import router as threads_router
from ai.bootstrap import RuntimeOverrides, build_lifespan
from ai.shared.config.settings import Settings, load_settings


def create_app(
    settings: Settings | None = None,
    *,
    overrides: RuntimeOverrides | None = None,
    **override_kwargs: Any,
) -> FastAPI:
    """Create the FastAPI application and attach routers."""
    if overrides is not None and override_kwargs:
        raise TypeError("Pass either 'overrides' or keyword overrides, not both.")
    active_overrides = overrides if overrides is not None else RuntimeOverrides(**override_kwargs)
    active_settings = settings or load_settings()
    app = FastAPI(
        title="AI Multi-Agent Service",
        version="0.1.0",
        lifespan=build_lifespan(active_settings, overrides=active_overrides),
    )
    for router in (
        health_router,
        chat_router,
        threads_router,
        knowledge_router,
        agents_router,
    ):
        app.include_router(router)
    return app


app = create_app()
