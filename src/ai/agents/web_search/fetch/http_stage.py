"""Fetch stage doing SSRF-guarded HTTP with bounded redirect hops."""

from urllib.parse import urljoin

import httpx

from ai.agents.web_search.fetch.contracts import (
    BlockedReason,
    FetchRequest,
    FetchResult,
)
from ai.agents.web_search.fetch.extractor import (
    extract_markdown_from_html,
)
from ai.agents.web_search.fetch.security import (
    DEFAULT_BLOCKED_DOMAINS,
    DnsResolver,
    SsrfBlockedError,
    detect_bot_challenge,
    validate_safe_public_url,
)
from ai.agents.web_search.search.scope import url_matches_domain_scope

_ALLOWED_TEXT_MIME_PREFIXES = (
    "text/html",
    "text/plain",
    "text/markdown",
    "application/xhtml+xml",
    "application/json",
)
_REDIRECT_STATUSES: frozenset[int] = frozenset({301, 302, 303, 307, 308})


class SafeHttpFetchStage:
    """Stage 3: Direct HTTP fetch with per-hop SSRF validation, byte cap & bot detection."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 10.0,
        max_fetch_bytes: int = 2 * 1024 * 1024,
        max_redirects: int = 4,
        user_agent: str = "FluxResearchAI/0.1",
        blocked_domains: frozenset[str] = DEFAULT_BLOCKED_DOMAINS,
        dns_resolver: DnsResolver | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._timeout_seconds = timeout_seconds
        self._max_fetch_bytes = max_fetch_bytes
        self._max_redirects = max_redirects
        self._user_agent = user_agent
        self._blocked_domains = blocked_domains
        self._dns_resolver = dns_resolver
        self._http_client = http_client

    @property
    def name(self) -> str:
        return "safe_http"

    def _is_domain_blocked(self, url: str, request: FetchRequest) -> bool:
        if not url_matches_domain_scope(
            url,
            allowed_domains=(),
            blocked_domains=tuple(self._blocked_domains),
        ):
            return True
        if request.allowed_domains or request.blocked_domains:
            return not url_matches_domain_scope(
                url,
                allowed_domains=request.allowed_domains,
                blocked_domains=request.blocked_domains,
            )
        return False

    async def _send_single_hop(
        self, url: str, *, client: httpx.AsyncClient
    ) -> tuple[httpx.Response, bytes]:
        headers = {
            "User-Agent": self._user_agent,
            "Accept": "text/html,application/xhtml+xml,text/plain,text/markdown;q=0.9",
        }
        async with client.stream(
            "GET",
            url,
            headers=headers,
            follow_redirects=False,
            timeout=self._timeout_seconds,
        ) as resp:
            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.aiter_bytes():
                if total + len(chunk) > self._max_fetch_bytes:
                    needed = self._max_fetch_bytes - total
                    if needed > 0:
                        chunks.append(chunk[:needed])
                    break
                chunks.append(chunk)
                total += len(chunk)
            return resp, b"".join(chunks)

    def _blocked(
        self,
        request: FetchRequest,
        final_url: str,
        reason: BlockedReason,
    ) -> FetchResult:
        return FetchResult(
            url=request.url,
            final_url=final_url,
            stage_resolved=self.name,
            is_blocked=True,
            blocked_reason=reason,
            identifiers=request.identifiers,
        )

    async def _validate_url(
        self,
        url: str,
        request: FetchRequest,
    ) -> tuple[str | None, FetchResult | None]:
        if self._is_domain_blocked(url, request):
            return None, self._blocked(request, url, "domain_blocked")
        try:
            validated = await validate_safe_public_url(url, dns_resolver=self._dns_resolver)
        except SsrfBlockedError:
            return None, self._blocked(request, url, "ssrf_blocked")
        return validated, None

    async def try_fetch(self, request: FetchRequest) -> FetchResult | None:
        validated_url, blocked = await self._validate_url(request.url.strip(), request)
        if blocked is not None or validated_url is None:
            return blocked

        if self._http_client is not None:
            return await self._execute_hops(request, validated_url, client=self._http_client)

        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            return await self._execute_hops(request, validated_url, client=client)

    async def _execute_hops(
        self,
        request: FetchRequest,
        start_url: str,
        *,
        client: httpx.AsyncClient,
    ) -> FetchResult:
        current_url = start_url
        response: httpx.Response | None = None
        raw_bytes = b""

        for _ in range(self._max_redirects + 1):
            try:
                response, raw_bytes = await self._send_single_hop(current_url, client=client)
            except httpx.HTTPError:
                return self._blocked(request, current_url, "network_error")

            if response.status_code in _REDIRECT_STATUSES:
                location = response.headers.get("location")
                if not location:
                    break
                next_url = urljoin(current_url, location)
                validated_next, blocked = await self._validate_url(next_url, request)
                if blocked is not None or validated_next is None:
                    return blocked or self._blocked(request, next_url, "ssrf_blocked")
                current_url = validated_next
                continue
            break

        if response is None or response.status_code in _REDIRECT_STATUSES:
            return self._blocked(request, current_url, "network_error")

        body_text = raw_bytes.decode("utf-8", errors="replace")
        blocked_reason = detect_bot_challenge(body_text, response.status_code)
        if blocked_reason is not None:
            return self._blocked(request, current_url, blocked_reason)

        if response.status_code >= 400:
            return self._blocked(request, current_url, "network_error")

        content_type = (
            response.headers.get("content-type", "text/html").split(";", 1)[0].strip().lower()
        )
        if content_type and not any(
            content_type.startswith(p) for p in _ALLOWED_TEXT_MIME_PREFIXES
        ):
            return self._blocked(request, current_url, "unsupported_content")

        if content_type in {"text/plain", "text/markdown", "application/json"}:
            title = request.url
            markdown = body_text[:12_000].strip()
        else:
            title, markdown = extract_markdown_from_html(body_text)

        return FetchResult(
            url=request.url,
            final_url=current_url,
            title=title or request.url,
            markdown=markdown,
            stage_resolved=self.name,
            is_blocked=False,
            identifiers=request.identifiers,
        )
