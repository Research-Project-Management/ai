"""Domain contracts, exceptions, and protocols for Global Resilience Shield."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class CircuitState(StrEnum):
    """Finite state machine states for a resilience Circuit Breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised immediately when calls are short-circuited by an OPEN breaker."""

    def __init__(self, service_name: str, retry_after_seconds: float) -> None:
        self.service_name = service_name
        self.retry_after_seconds = max(0.0, retry_after_seconds)
        super().__init__(
            f"Circuit breaker '{service_name}' is OPEN; retry after {self.retry_after_seconds:.1f}s."
        )


class CircuitBreakerConfig(BaseModel):
    """Immutable configuration controlling Circuit Breaker trip and recovery thresholds."""

    model_config = ConfigDict(frozen=True)

    failure_threshold: int = Field(default=5, ge=1)
    recovery_timeout_seconds: float = Field(default=30.0, gt=0.0)
    half_open_max_calls: int = Field(default=1, ge=1)


class ToolCache(Protocol):
    """Port defining key-value TTL cache operations for idempotent tool outputs."""

    async def get(self, key: str) -> Any | None: ...

    async def set(self, key: str, value: Any, *, ttl_seconds: int = 300) -> None: ...

    async def delete(self, key: str) -> None: ...
