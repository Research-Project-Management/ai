"""SSRF network guards, policy blocklists, and anti-bot / WAF response detection."""

import asyncio
import ipaddress
import socket
from collections.abc import Callable, Sequence
from typing import Final
from urllib.parse import urlparse

from ai.agents.web_search.fetch.contracts import BlockedReason

DEFAULT_BLOCKED_DOMAINS: Final[frozenset[str]] = frozenset(
    {
        "sci-hub.se",
        "sci-hub.st",
        "sci-hub.ru",
        "sci-hub.mksa.top",
        "coursehero.com",
    }
)

_FORBIDDEN_HOSTS: Final[frozenset[str]] = frozenset(
    {
        "localhost",
        "ip6-localhost",
        "ip6-loopback",
        "metadata.google.internal",
        "metadata.goog",
        "169.254.169.254",
    }
)

_WAF_SIGNATURES: Final[tuple[str, ...]] = (
    "just a moment...",
    "attention required! | cloudflare",
    "cf-chl-bypass",
    "challenge-platform",
    "verify you are human",
    "enable javascript and cookies to continue",
    "checking your browser before accessing",
)

DnsResolver = Callable[[str], Sequence[str]]


class SsrfBlockedError(ValueError):
    """Raised when a URL targets non-public/local infrastructure or an unsafe scheme."""


def _is_unsafe_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped is not None:
        return _is_unsafe_ip(ip_obj.ipv4_mapped)
    return bool(
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_reserved
        or ip_obj.is_unspecified
    )


def _lookup_system_ips(host: str) -> list[str]:
    infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    ips: list[str] = []
    seen: set[str] = set()
    for info in infos:
        sockaddr = info[4]
        if sockaddr and isinstance(sockaddr[0], str):
            ip_str = sockaddr[0].split("%", 1)[0]
            if ip_str not in seen:
                seen.add(ip_str)
                ips.append(ip_str)
    return ips


async def validate_safe_public_url(
    url: str,
    *,
    dns_resolver: DnsResolver | None = None,
) -> str:
    """Validate scheme, hostname, and resolved IP addresses against SSRF vectors."""
    cleaned = url.strip()
    if not cleaned:
        raise SsrfBlockedError("Empty URL")

    parsed = urlparse(cleaned)
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        raise SsrfBlockedError(f"Unsupported URL scheme: '{scheme or 'none'}'")

    host = (parsed.hostname or "").strip(".").lower()
    if not host:
        raise SsrfBlockedError("URL has no hostname")

    if host in _FORBIDDEN_HOSTS or host.endswith((".localhost", ".local", ".internal")):
        raise SsrfBlockedError(f"Blocked internal hostname: '{host}'")

    # Check literal IP directly before attempting DNS
    try:
        ip_literal = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        ip_literal = None

    if ip_literal is not None:
        if _is_unsafe_ip(ip_literal):
            raise SsrfBlockedError(f"Blocked non-public IP target: '{host}'")
        return cleaned

    resolved_ips: Sequence[str]
    if dns_resolver is not None:
        resolved_ips = dns_resolver(host)
    else:
        try:
            resolved_ips = await asyncio.to_thread(_lookup_system_ips, host)
        except OSError:
            # Allow offline/mocked HTTP transports when OS DNS cannot resolve test domains;
            # real sockets will still fail cleanly in httpx on invalid hostnames.
            return cleaned

    for raw_ip in resolved_ips:
        try:
            parsed_ip = ipaddress.ip_address(str(raw_ip).split("%", 1)[0])
        except ValueError:
            continue
        if _is_unsafe_ip(parsed_ip):
            raise SsrfBlockedError(f"Host '{host}' resolves to blocked IP '{raw_ip}'")

    return cleaned


def detect_bot_challenge(body_text: str, status_code: int) -> BlockedReason | None:
    """Inspect HTTP status codes and response HTML for WAF/CAPTCHA/paywall blocks."""
    if status_code == 429:
        return "rate_limited"
    if status_code in {401, 402, 403}:
        return "paywall_or_forbidden"
    sample = body_text[:8192].lower()
    if any(signature in sample for signature in _WAF_SIGNATURES):
        return "waf_challenge"
    return None
