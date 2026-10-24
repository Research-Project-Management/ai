"""Contracts and configuration value objects for the `rag_agent` reasoning workflow."""

from typing import Final

from pydantic import BaseModel, ConfigDict, Field

ABSTENTION_RESPONSE: Final[str] = (
    "Không tìm thấy thông tin phù hợp trong tài liệu của Space hiện tại để trả lời chính xác câu hỏi này."
)


class RagAgentConfig(BaseModel):
    """Immutable bounded configuration for the Corrective RAG subgraph."""

    model_config = ConfigDict(frozen=True)

    max_attempts: int = Field(default=2, ge=1, le=3)
    max_queries: int = Field(default=3, ge=1, le=4)
    min_passages: int = Field(default=1, ge=1)
    timeout_seconds: float = Field(default=45.0, gt=0.0, le=300.0)


class QueryTransformResult(BaseModel):
    """Structured query rewrite and bilingual expansion plan."""

    model_config = ConfigDict(frozen=True)

    standalone_query: str = Field(..., min_length=1)
    search_queries: tuple[str, ...] = ()
