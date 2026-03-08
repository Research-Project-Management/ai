"""Health check endpoint."""

from fastapi import APIRouter

from app.models import HealthStatus
from app.rag.store import check_qdrant_health

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthStatus)
async def health_check():
    """Check system health — LLM, Qdrant, Redis status."""
    services = {}

    # Qdrant
    qdrant_ok = await check_qdrant_health()
    services["qdrant"] = "ok" if qdrant_ok else "unavailable"

    # Redis
    try:
        import redis.asyncio as aioredis
        from app.config import get_settings

        settings = get_settings()
        r = aioredis.from_url(settings.redis_url)
        await r.ping()
        services["redis"] = "ok"
        await r.aclose()
    except Exception:
        services["redis"] = "unavailable"

    # Overall
    overall = "ok" if all(v == "ok" for v in services.values()) else "degraded"

    return HealthStatus(status=overall, services=services)
