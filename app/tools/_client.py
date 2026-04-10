"""
Internal API client — FLux-AI → RPM-BE.

Uses ``contextvars`` so each async request carries its own user context,
making it safe for concurrent agent sessions.

Usage::

    from app.tools._client import flux_api, set_user_context

    set_user_context("user_id_here", "workspace_id_here")
    result = await flux_api.get("/api/workspace/my-workspace")
"""

from __future__ import annotations

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


def set_user_context(
    user_id: str,
    workspace_id: str = "",
    project_id: str = "",
) -> None:
    """Set the user context for the current async task."""
    _current_user_id.set(user_id)
    _current_workspace_id.set(workspace_id)
    _current_project_id.set(project_id)


def get_user_id() -> str:
    return _current_user_id.get()


def get_workspace_id() -> str:
    return _current_workspace_id.get()


def get_project_id() -> str:
    return _current_project_id.get()


class FluxAPIClient:
    """HTTP client that calls RPM-BE with internal auth headers.

    The ``X-Internal-Key`` header authenticates the request as coming
    from FLux-AI, and ``X-User-Id`` specifies which user the agent is
    acting on behalf of. RPM-BE's ``isAuthenticated`` middleware
    recognises these headers and impersonates the user.
    """

    def __init__(self):
        settings = get_settings()
        self.base_url = settings.rpm_be_url.rstrip("/")
        self.internal_key = settings.internal_api_key

    async def request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict | list | str:
        """Make an authenticated request to RPM-BE."""
        user_id = _current_user_id.get()
        if not user_id:
            raise RuntimeError(
                "No user context set. Call set_user_context() before using tools."
            )

        headers = {
            "X-Internal-Key": self.internal_key,
            "X-User-Id": user_id,
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}{path}"
        logger.debug("API %s %s (user=%s)", method, path, user_id[:8])

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.request(method, url, headers=headers, **kwargs)

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

    async def get(self, path: str, **kw: Any):
        return await self.request("GET", path, **kw)

    async def post(self, path: str, **kw: Any):
        return await self.request("POST", path, **kw)

    async def put(self, path: str, **kw: Any):
        return await self.request("PUT", path, **kw)

    async def delete(self, path: str, **kw: Any):
        return await self.request("DELETE", path, **kw)


# Singleton
flux_api = FluxAPIClient()
