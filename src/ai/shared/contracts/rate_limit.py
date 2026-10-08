"""Rate limiting contracts (domain interface and immutable decision result)."""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class RateLimitResult(BaseModel):
    """Immutable outcome of a sliding-window rate limit check."""

    model_config = ConfigDict(frozen=True)

    allowed: bool
    limit: int = Field(..., ge=0)
    remaining: int = Field(..., ge=0)
    reset_after_seconds: int = Field(..., ge=0)
    reason: str = ""


@runtime_checkable
class RateLimiter(Protocol):
    """Port interface for scoped request rate limiters."""

    async def check(self, key: str) -> RateLimitResult:
        """Evaluate and record one request against the quota for `key`."""
        ...
