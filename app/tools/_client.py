"""
Internal API client — FLux-AI → RPM-BE.

Uses ``contextvars`` so each async request carries its own user context,
making it safe for concurrent agent sessions.

The underlying HTTP client is a shared persistent httpx.AsyncClient with
connection pooling — avoiding the TCP handshake overhead of creating a
new client per tool call. This significantly improves throughput when
multiple users are chatting simultaneously.
"""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

# Per-request context — set by the agent endpoint before running the agent
_current_user_id: ContextVar[str] = ContextVar("current_user_id", default="")
_current_workspace_id: ContextVar[str] = ContextVar("current_workspace_id", default="")
_current_project_id: ContextVar[str] = ContextVar("current_project_id", default="")
_current_trace_id: ContextVar[str] = ContextVar("current_trace_id", default="-")
# Raw user message text — read by _guards.py for confirmation detection
_user_message: ContextVar[str] = ContextVar("_user_message", default="")


def set_user_context(
    user_id: str,
    workspace_id: str = "",
    project_id: str = "",
) -> None:
    """Set the user context for the current async task."""
    _current_user_id.set(user_id)
    _current_workspace_id.set(workspace_id)
    _current_project_id.set(project_id)


def set_trace_id(trace_id: str) -> None:
    """Set the distributed trace ID for the current async task."""
    _current_trace_id.set(trace_id)


def set_user_message(text: str) -> None:
    """Set the raw user message text for the current async task.
    Used by the server-side confirmation guard in _guards.py."""
    _user_message.set(text)


def get_trace_id() -> str:
    return _current_trace_id.get()


def get_user_id() -> str:
    return _current_user_id.get()


def get_workspace_id() -> str:
    return _current_workspace_id.get()


def get_project_id() -> str:
    return _current_project_id.get()


class FluxAPIClient:
    """HTTP client that calls RPM-BE with internal auth headers.

    Uses a shared persistent httpx.AsyncClient with connection pooling.
    This avoids the TCP handshake overhead of creating a new client per
    tool call, significantly improving throughput under concurrent load.

    Thread-safety: httpx.AsyncClient is safe to share across coroutines.
    User context is isolated via ContextVar (per asyncio Task).

    Rate-limit handling: automatically retries on HTTP 429 / 503 with
    exponential backoff (up to 3 attempts).
    """

    # Connection limits: up to 50 concurrent connections, 20 per host
    _LIMITS = httpx.Limits(
        max_connections=50,
        max_keepalive_connections=20,
        keepalive_expiry=30,
    )
    _TIMEOUT = httpx.Timeout(30.0, connect=5.0)
    _MAX_RETRIES = 3
    _RETRY_STATUSES = {429, 503}

    def __init__(self):
        settings = get_settings()
        self.base_url = settings.rpm_be_url.rstrip("/")
        self.internal_key = settings.internal_api_key
        # Shared persistent client — created lazily on first use
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return the shared client, creating it if necessary."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                limits=self._LIMITS,
                timeout=self._TIMEOUT,
            )
        return self._client

    async def close(self) -> None:
        """Gracefully close the connection pool (call on app shutdown)."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            logger.info("FluxAPIClient connection pool closed.")

    async def request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict | list | str:
        """Make an authenticated request to RPM-BE with retry on rate limit."""
        user_id = _current_user_id.get()
        if not user_id:
            logger.warning(
                "No user context set for %s %s — call set_user_context() before using tools.",
                method, path,
            )
            return {"error": True, "status": 401, "message": "No user context set"}

        trace_id = _current_trace_id.get()
        headers = {
            "X-Internal-Key": self.internal_key,
            "X-User-Id": user_id,
            "X-Trace-Id": trace_id,
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}{path}"
        logger.debug("[%s] API %s %s (user=%s)", trace_id, method, path, user_id[:8])

        client = self._get_client()
        last_error: Exception | None = None

        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                resp = await client.request(method, url, headers=headers, **kwargs)

                # Retry on rate limit / transient errors
                if resp.status_code in self._RETRY_STATUSES and attempt < self._MAX_RETRIES:
                    wait = 2 ** attempt  # 2s, 4s
                    logger.warning(
                        "API %s %s → %d (rate limit). Retry %d/%d in %ds…",
                        method, path, resp.status_code, attempt, self._MAX_RETRIES, wait,
                    )
                    await asyncio.sleep(wait)
                    continue

                if resp.status_code >= 400:
                    error_text = resp.text[:500]
                    logger.warning(
                        "API error %s %s → %d: %s",
                        method, path, resp.status_code, error_text,
                    )
                    return {
                        "error": True,
                        "status": resp.status_code,
                        "message": error_text,
                    }

                if resp.status_code == 204:
                    return {"success": True}

                try:
                    return resp.json()
                except Exception:
                    return {"raw": resp.text[:1000]}

            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt < self._MAX_RETRIES:
                    await asyncio.sleep(2 ** attempt)
                    continue
            except Exception as exc:
                logger.error("API request failed: %s %s → %s", method, path, exc)
                return {"error": True, "status": 500, "message": str(exc)}

        return {
            "error": True,
            "status": 504,
            "message": f"Timeout after {self._MAX_RETRIES} retries: {last_error}",
        }

    async def get(self, path: str, **kw: Any):
        return await self.request("GET", path, **kw)

    async def post(self, path: str, **kw: Any):
        return await self.request("POST", path, **kw)

    async def put(self, path: str, **kw: Any):
        return await self.request("PUT", path, **kw)

    async def delete(self, path: str, **kw: Any):
        return await self.request("DELETE", path, **kw)


# Shared singleton — connection pool is reused across ALL requests and users
flux_api = FluxAPIClient()
