"""Domain allow/block scoping shared by every search provider."""

from urllib.parse import urlparse


def apply_site_operators(
    raw_query: str,
    *,
    allowed_domains: tuple[str, ...],
    blocked_domains: tuple[str, ...],
) -> str:
    """Append search engine `site:` / `-site:` clauses when domain filters are active."""
    base = raw_query.strip()
    if allowed_domains:
        clauses = " OR ".join(f"site:{domain}" for domain in allowed_domains[:8])
        return f"({base}) ({clauses})" if len(allowed_domains) > 1 else f"{base} {clauses}"
    if blocked_domains:
        exclusions = " ".join(f"-site:{domain}" for domain in blocked_domains[:8])
        return f"{base} {exclusions}"
    return base


def _host_matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith(f".{domain}")


def url_matches_domain_scope(
    url: str,
    *,
    allowed_domains: tuple[str, ...],
    blocked_domains: tuple[str, ...],
) -> bool:
    """Enforce allowed/blocked domain boundaries independently of engine quirks."""
    host = (urlparse(url).hostname or "").strip(".").lower()
    if not host:
        return False
    if allowed_domains:
        return any(_host_matches(host, domain) for domain in allowed_domains)
    if blocked_domains:
        return not any(_host_matches(host, domain) for domain in blocked_domains)
    return True
