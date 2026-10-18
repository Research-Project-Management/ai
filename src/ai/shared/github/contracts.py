"""Contracts and DTOs for GitHub artifact fetching and data extraction."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class GitHubArtifactError(RuntimeError):
    """Domain exception raised when failing to retrieve or decode GitHub artifacts."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class GitHubArtifactContent(BaseModel):
    """Extracted text artifact with verified commit provenance."""

    model_config = ConfigDict(frozen=True)

    repo: str
    path: str
    ref: str
    commit_sha: str
    content: str
    format: Literal["csv", "tsv", "json"]
    size_bytes: int = 0
    url: str = Field(default="")


class GitHubArtifactGateway(Protocol):
    """Port defining artifact retrieval operations against GitHub repositories."""

    async def fetch_artifact(
        self,
        *,
        repo: str,
        path: str,
        ref: str = "main",
        auth_token: str | None = None,
    ) -> GitHubArtifactContent: ...
