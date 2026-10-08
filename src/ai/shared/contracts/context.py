"""Verified workspace scope context injected from the backend gateway."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class SpaceContext(BaseModel):
    """Immutable scope context aligned 1:1 with Flux `Space` (`personal` | `project`).

    A Flux user has a personal `Space` (`type=personal`), and every project has a
    dedicated `Space` (`type=project`). Downstream AI workflows are therefore scoped
    by `space_id` without duplicating redundant `user_id` or `project_id` fields.
    """

    model_config = ConfigDict(frozen=True)

    space_id: str = Field(..., min_length=1)
    chat_id: str = Field(..., min_length=1)
    page_id: str | None = None
    document_ids: tuple[str, ...] = ()
    delegation_token: SecretStr = SecretStr("")
    trace_id: str = ""

    def to_configurable(self) -> dict[str, Any]:
        """Build JSON-safe LangGraph `configurable` metadata (excludes bearer secrets)."""
        return {
            "thread_id": self.chat_id,
            "space_id": self.space_id,
            "chat_id": self.chat_id,
            "page_id": self.page_id,
            "document_ids": list(self.document_ids),
            "trace_id": self.trace_id,
        }
