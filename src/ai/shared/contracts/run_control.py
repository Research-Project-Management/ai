"""Contracts for execution usage telemetry and active run cancellation."""

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class RunUsage(BaseModel):
    """Immutable token and latency accounting for an orchestration run."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)

    @classmethod
    def from_message(cls, message: Any) -> "RunUsage":
        """Extract token usage metadata from a LangChain message object."""
        meta = getattr(message, "usage_metadata", None)
        if not isinstance(meta, dict):
            return cls()
        input_tokens = max(0, int(meta.get("input_tokens") or 0))
        output_tokens = max(0, int(meta.get("output_tokens") or 0))
        total_tokens = max(0, int(meta.get("total_tokens") or (input_tokens + output_tokens)))
        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def merge(self, other: "RunUsage") -> "RunUsage":
        """Accumulate token counts across multiple LLM invocations."""
        return RunUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            latency_ms=max(self.latency_ms, other.latency_ms),
        )

    def with_latency(self, latency_ms: float) -> "RunUsage":
        """Return a copy with the measured end-to-end latency in milliseconds."""
        return RunUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
            latency_ms=max(0.0, round(latency_ms, 2)),
        )


@runtime_checkable
class RunCancellationRegistry(Protocol):
    """Port interface for signalling cooperative cancellation of active runs."""

    async def clear(self, space_id: str, thread_id: str) -> None:
        """Clear any prior cancellation flag before starting a new run."""
        ...

    async def request_cancel(self, space_id: str, thread_id: str) -> None:
        """Signal that any in-flight run for `(space_id, thread_id)` should abort."""
        ...

    async def is_cancelled(self, space_id: str, thread_id: str) -> bool:
        """Check whether cancellation has been requested for `(space_id, thread_id)`."""
        ...
