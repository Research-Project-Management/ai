"""HTTP adapter fetching experiment artifacts via GitHub REST API."""

from __future__ import annotations

import base64
import hashlib
import re
from typing import Literal

import httpx

from ai.shared.github.contracts import (
    GitHubArtifactContent,
    GitHubArtifactError,
    GitHubArtifactGateway,
)
from ai.shared.resilience.cache import build_cache_key
from ai.shared.resilience.circuit_breaker import CircuitBreaker
from ai.shared.resilience.contracts import CircuitOpenError, ToolCache

_SHA_REGEX = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


def _detect_format(path: str, raw_content: str) -> Literal["csv", "tsv", "json"]:
    lower = path.lower()
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".tsv"):
        return "tsv"
    if lower.endswith(".json"):
        return "json"
    stripped = raw_content.strip()
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        return "json"
    if "\t" in stripped:
        return "tsv"
    return "csv"


def _is_transient_failure(exc: GitHubArtifactError) -> bool:
    return exc.status_code is None or exc.status_code >= 500 or exc.status_code == 429


class HttpGitHubArtifactClient(GitHubArtifactGateway):
    """Adapter fetching raw file artifacts and verified commit SHAs using GitHub REST API."""

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient | None = None,
        default_token: str | None = None,
        timeout_seconds: float = 15.0,
        circuit_breaker: CircuitBreaker | None = None,
        cache: ToolCache | None = None,
        cache_ttl_seconds: int = 300,
    ) -> None:
        self._http_client = http_client
        self._default_token = default_token
        self._timeout_seconds = timeout_seconds
        self._circuit_breaker = circuit_breaker
        self._cache = cache
        self._cache_ttl_seconds = cache_ttl_seconds

    async def _send(
        self,
        client: httpx.AsyncClient,
        url: str,
        token: str | None,
    ) -> httpx.Response:
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Flux-Research-Assistant",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            return await client.get(url, headers=headers, timeout=self._timeout_seconds)
        except (httpx.TimeoutException, OSError) as exc:
            raise GitHubArtifactError(f"Network error querying GitHub: {exc}") from exc

    async def fetch_artifact(
        self,
        *,
        repo: str,
        path: str,
        ref: str = "main",
        auth_token: str | None = None,
    ) -> GitHubArtifactContent:
        """Fetch file contents and resolve verified commit SHA."""
        clean_repo = repo.strip().strip("/")
        clean_path = path.strip().lstrip("/")
        clean_ref = ref.strip() or "main"
        token = auth_token or self._default_token

        if not clean_repo or "/" not in clean_repo:
            raise GitHubArtifactError(
                f"Invalid GitHub repository identifier '{repo}'. Expected 'owner/repo' format."
            )
        if not clean_path:
            raise GitHubArtifactError("File path cannot be empty.")

        auth_scope = (
            hashlib.sha256(token.encode("utf-8")).hexdigest()[:12] if token else "public"
        )
        cache_key = build_cache_key(
            "github_artifact",
            {
                "repo": clean_repo,
                "path": clean_path,
                "ref": clean_ref,
                "auth": auth_scope,
            },
        )
        if self._cache is not None:
            cached = await self._cache.get(cache_key)
            if isinstance(cached, dict):
                return GitHubArtifactContent.model_validate(cached)

        if self._circuit_breaker is not None:
            try:
                await self._circuit_breaker.allow_request()
            except CircuitOpenError as exc:
                raise GitHubArtifactError(str(exc), status_code=503) from exc

        contents_url = (
            f"https://api.github.com/repos/{clean_repo}/contents/{clean_path}?ref={clean_ref}"
        )

        async def _do_fetch(client: httpx.AsyncClient) -> GitHubArtifactContent:
            res = await self._send(client, contents_url, token)
            if res.status_code == 404:
                raise GitHubArtifactError(
                    f"Artifact '{clean_path}' in repo '{clean_repo}' (ref '{clean_ref}') was not found.",
                    status_code=404,
                )
            if res.status_code in (401, 403):
                raise GitHubArtifactError(
                    f"GitHub authorization error or rate limit exceeded ({res.status_code}) for '{clean_repo}'.",
                    status_code=res.status_code,
                )
            if res.status_code >= 400:
                raise GitHubArtifactError(
                    f"GitHub API returned error {res.status_code} for '{clean_repo}/{clean_path}'.",
                    status_code=res.status_code,
                )

            data = res.json()
            if not isinstance(data, dict) or "content" not in data:
                raise GitHubArtifactError(
                    f"Target path '{clean_path}' is not a file or returned invalid payload."
                )

            # Decode base64
            encoding = data.get("encoding", "")
            raw_b64 = str(data.get("content") or "")
            if encoding == "base64":
                try:
                    decoded_bytes = base64.b64decode(raw_b64.replace("\n", ""))
                    decoded_text = decoded_bytes.decode("utf-8")
                except ValueError as exc:  # binascii.Error and UnicodeDecodeError subclass it
                    raise GitHubArtifactError(f"Failed to decode base64 file content: {exc}") from exc
            else:
                decoded_text = raw_b64

            # Resolve actual commit SHA
            resolved_sha = clean_ref
            if not _SHA_REGEX.match(clean_ref):
                # Query commit endpoint to get true commit SHA
                commits_url = (
                    f"https://api.github.com/repos/{clean_repo}/commits/{clean_ref}"
                )
                commit_res = await self._send(client, commits_url, token)
                if commit_res.status_code == 200:
                    commit_data = commit_res.json()
                    if isinstance(commit_data, dict) and "sha" in commit_data:
                        resolved_sha = str(commit_data["sha"])
                else:
                    # Fallback to content blob sha if commit lookup fails
                    resolved_sha = str(data.get("sha") or clean_ref)

            file_format = _detect_format(clean_path, decoded_text)
            html_url = str(
                data.get("html_url")
                or f"https://github.com/{clean_repo}/blob/{resolved_sha}/{clean_path}"
            )

            return GitHubArtifactContent(
                repo=clean_repo,
                path=clean_path,
                ref=clean_ref,
                commit_sha=resolved_sha,
                content=decoded_text,
                format=file_format,
                size_bytes=int(data.get("size") or len(decoded_text.encode("utf-8"))),
                url=html_url,
            )

        try:
            if self._http_client is not None:
                result = await _do_fetch(self._http_client)
            else:
                async with httpx.AsyncClient() as client:
                    result = await _do_fetch(client)
        except GitHubArtifactError as exc:
            if self._circuit_breaker is not None and _is_transient_failure(exc):
                await self._circuit_breaker.record_failure()
            raise

        if self._circuit_breaker is not None:
            await self._circuit_breaker.record_success()
        if self._cache is not None:
            await self._cache.set(
                cache_key,
                result.model_dump(mode="json"),
                ttl_seconds=self._cache_ttl_seconds,
            )
        return result
