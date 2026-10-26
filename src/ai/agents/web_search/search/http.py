"""Single HTTP request path for search providers (shared client or one-shot client)."""

from typing import Any

import httpx


async def http_request(
    *,
    method: str,
    url: str,
    http_client: httpx.AsyncClient | None,
    timeout_seconds: float,
    headers: dict[str, str],
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> httpx.Response:
    if http_client is not None:
        return await http_client.request(
            method,
            url,
            params=params,
            json=json_body,
            headers=headers,
            timeout=timeout_seconds,
        )
    async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True) as client:
        return await client.request(
            method,
            url,
            params=params,
            json=json_body,
            headers=headers,
        )
