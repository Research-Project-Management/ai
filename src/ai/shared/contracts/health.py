"""Infrastructure health check contracts."""

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class ComponentStatus(BaseModel):
    """Immutable health report for a single dependency or subsystem."""

    model_config = ConfigDict(frozen=True)

    name: str
    status: Literal["ok", "degraded", "disabled"]
    required: bool = True
    latency_ms: float | None = None
    detail: str = ""


@runtime_checkable
class HealthProbe(Protocol):
    """Port interface for deep readiness/health checks."""

    @property
    def name(self) -> str:
        """Identifier returned in HealthResponse.components."""
        ...

    @property
    def required(self) -> bool:
        """Whether a degraded status marks overall service health as degraded."""
        ...

    async def check(self) -> ComponentStatus:
        """Execute the probe and return its status."""
        ...
