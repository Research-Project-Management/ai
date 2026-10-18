"""Shared async HTTP transport for Flux Backend clients: JWT delegation and error translation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import httpx

from ai.shared.contracts.auth import get_delegation_token

GatewayErrorFactory = Callable[[int | None, str], Exception]


def encode_seg(value: str) -> str:
    """URL-encode a single path parameter segment cleanly."""
    return quote(value.strip(), safe="")


def unwrap_envelope(payload: Any) -> Any:
    """Unwrap NestJS `TransformInterceptor` `{ success, data, timestamp }` envelopes."""
    current = payload
    for _ in range(2):
        if (
            isinstance(current, dict)
            and "data" in current
            and current["data"] is not None
            and ("success" in current or "timestamp" in current)
        ):
            current = current["data"]
        else:
            break
    return current


def extract_error_message(response: httpx.Response) -> str:
    """Extract clean diagnostic text from NestJS `GlobalExceptionFilter` or plain JSON errors."""
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        err_obj = payload.get("error")
        if isinstance(err_obj, dict) and err_obj.get("message"):
            return str(err_obj["message"])[:300]
        msg = payload.get("message") or payload.get("detail") or err_obj
        if isinstance(msg, list):
            return "; ".join(str(part) for part in msg)[:300]
        if msg:
            return str(msg)[:300]
    text = response.text.strip()
    return text[:300] if text else f"HTTP {response.status_code}"


class BackendTransport:
    """One request path for every Backend gateway: auth, timeouts, envelope, typed errors.

    `service` names the Backend area ("Pages", "Library", "Project") in error messages and
    `error_factory` builds that gateway's own error type, so callers keep their error contract.
    """

    def __init__(
        self,
        *,
        base_url: str,
        service: str,
        error_factory: GatewayErrorFactory,
        timeout_seconds: float = 15.0,
        http_client: httpx.AsyncClient | None = None,
        token_provider: Callable[[], str] = get_delegation_token,
        internal_api_key: str = "",
    ) -> None:
        self._base_url = base_url.strip().rstrip("/")
        self._service = service
        self._error = error_factory
        self._timeout_seconds = timeout_seconds
        self._http_client = http_client
        self._token_provider = token_provider
        self._internal_api_key = internal_api_key.strip()

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        if not self._base_url:
            raise self._error(None, f"{self._service} backend URL is not configured")

        token = self._token_provider().strip()
        if not token:
            raise self._error(
                401, f"Missing user authorization token for Flux {self._service} access"
            )

        clean_params = {k: v for k, v in (params or {}).items() if v is not None and v != ""}
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        if self._internal_api_key:
            headers["X-Internal-Key"] = self._internal_api_key

        url = f"{self._base_url}{path}"
        try:
            if self._http_client is not None:
                response = await self._http_client.request(
                    method,
                    url,
                    params=clean_params or None,
                    json=json_body,
                    headers=headers,
                    timeout=self._timeout_seconds,
                )
            else:
                async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                    response = await client.request(
                        method,
                        url,
                        params=clean_params or None,
                        json=json_body,
                        headers=headers,
                    )
        except httpx.TimeoutException as exc:
            raise self._error(504, f"{self._service} request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise self._error(None, f"{self._service} connection failed: {exc}") from exc

        if response.status_code >= 400:
            detail = extract_error_message(response)
            raise self._error(
                response.status_code,
                f"Backend returned HTTP {response.status_code}: {detail}",
            )

        if not response.content:
            return {}
        try:
            return unwrap_envelope(response.json())
        except ValueError:
            return {"text": response.text}
