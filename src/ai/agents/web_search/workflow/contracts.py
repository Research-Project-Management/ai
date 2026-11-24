"""Immutable workflow contracts and defaults for `web_search_agent`."""

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

WEB_ABSTENTION_RESPONSE: Final[str] = (
    "Không tìm thấy đủ nguồn công khai hoặc học thuật đáng tin cậy đã xác thực cho truy vấn này."
)

SearchMode = Literal["direct_url", "scholar", "web", "hybrid"]
VALID_MODES: Final[frozenset[str]] = frozenset({"direct_url", "scholar", "web", "hybrid"})


class SearchPlan(BaseModel):
    """Structured query decomposition and channel routing output."""

    model_config = ConfigDict(frozen=True)

    standalone_query: str = Field(..., min_length=1)
    mode: SearchMode = "hybrid"
    sub_queries: tuple[str, ...] = ()


class WebSearchAgentConfig(BaseModel):
    """Tuning and guardrail knobs for the 4-node `web_search_agent` subgraph."""

    model_config = ConfigDict(frozen=True)

    max_rounds: int = Field(default=2, ge=1, le=4)
    max_sub_queries: int = Field(default=3, ge=1, le=5)
    min_evidence_count: int = Field(default=1, ge=1, le=10)
    top_k_evidence: int = Field(default=6, ge=1, le=20)
    max_deep_fetch_urls: int = Field(default=3, ge=0, le=10)
    max_results_per_provider: int = Field(default=6, ge=1, le=25)
    min_relevance_score: float = Field(default=0.05, ge=0.0, le=1.0)
    search_timeout_seconds: float = Field(default=10.0, gt=0.0, le=60.0)
    timeout_seconds: float = Field(default=60.0, gt=0.0, le=300.0)
