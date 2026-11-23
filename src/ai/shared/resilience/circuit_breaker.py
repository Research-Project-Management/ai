"""State-pattern Circuit Breaker and Registry protecting external tool dependencies."""

from __future__ import annotations

import asyncio
import time

from ai.shared.resilience.contracts import (
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
)


class CircuitBreaker:
    """Async-safe Circuit Breaker preventing cascading failures across external APIs."""

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        self.name = name
        self._config = config or CircuitBreakerConfig()
        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._opened_at: float | None = None
        self._half_open_in_flight: int = 0
        self._lock = asyncio.Lock()

    @property
    def config(self) -> CircuitBreakerConfig:
        return self._config

    @property
    def state(self) -> CircuitState:
        """Current effective state (auto-projects OPEN -> HALF_OPEN once recovery elapses)."""
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._config.recovery_timeout_seconds:
                return CircuitState.HALF_OPEN
        return self._state

    async def allow_request(self) -> None:
        """Verify request is allowed under current state; raises CircuitOpenError otherwise."""
        async with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - (self._opened_at or 0.0)
                if elapsed >= self._config.recovery_timeout_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_in_flight = 0
                else:
                    retry_after = self._config.recovery_timeout_seconds - elapsed
                    raise CircuitOpenError(self.name, retry_after)

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_in_flight >= self._config.half_open_max_calls:
                    raise CircuitOpenError(self.name, self._config.recovery_timeout_seconds)
                self._half_open_in_flight += 1

    async def record_success(self) -> None:
        """Reset failure counters and close the circuit."""
        async with self._lock:
            self._failure_count = 0
            self._half_open_in_flight = 0
            self._opened_at = None
            self._state = CircuitState.CLOSED

    async def record_failure(self) -> None:
        """Record a failure and trip the circuit to OPEN when thresholds are met."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._half_open_in_flight = 0
                self._failure_count = self._config.failure_threshold
                return

            if self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self._config.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._opened_at = time.monotonic()
                    self._half_open_in_flight = 0


class CircuitBreakerRegistry:
    """Registry managing isolated CircuitBreaker instances per downstream service."""

    def __init__(self, default_config: CircuitBreakerConfig | None = None) -> None:
        self._default_config = default_config or CircuitBreakerConfig()
        self._breakers: dict[str, CircuitBreaker] = {}

    def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Return existing breaker for `name` or register a new one."""
        clean_name = name.strip()
        if clean_name not in self._breakers:
            self._breakers[clean_name] = CircuitBreaker(clean_name, config or self._default_config)
        return self._breakers[clean_name]

    def snapshot(self) -> dict[str, str]:
        """Return a mapping of service name to current CircuitState string value."""
        return {name: breaker.state.value for name, breaker in self._breakers.items()}
