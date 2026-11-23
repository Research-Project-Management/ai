"""Request, response, and operational schemas aligned with Flux Spaces."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ai.agents.rag.ingestion.jobs import IngestionJobStatus
from ai.orchestrator.rewind import CheckpointSummary
from ai.shared.contracts.health import ComponentStatus
from ai.shared.contracts.knowledge import KnowledgeHit
from ai.shared.contracts.run_control import FeedbackRecord, RunUsage
from ai.shared.llm.catalog import ModelCatalogEntry
from ai.shared.resilience.mutations import RevertStepResult


class ChatMessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    """Inbound AI chat payload (stateful when `chat_id` set, stateless otherwise)."""

    model_config = ConfigDict(extra="ignore")

    messages: list[ChatMessageInput] = Field(..., min_length=1)
    space_id: str = Field(..., min_length=1)
    chat_id: str | None = None
    stateless: bool = False
    page_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    intent_hint: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class ResumeRequest(BaseModel):
    """HITL resume payload scoped by `space_id` and `chat_id`."""

    model_config = ConfigDict(extra="ignore")

    decision: Any
    space_id: str = Field(..., min_length=1)
    chat_id: str = Field(..., min_length=1)
    page_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    content: str
    route: str = "direct"
    status: Literal["completed", "interrupted"] = "completed"
    citations: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    interrupts: list[Any] = Field(default_factory=list)
    usage: RunUsage = Field(default_factory=RunUsage)
    stateless: bool = False
    trace_id: str = ""


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    app_env: str
    components: list[ComponentStatus] = Field(default_factory=list)


class ThreadStateResponse(BaseModel):
    chat_id: str
    space_id: str
    exists: bool
    status: Literal["empty", "idle", "interrupted"]
    message_count: int = 0
    route: str | None = None
    interrupts: list[Any] = Field(default_factory=list)


class ThreadCancelResponse(BaseModel):
    chat_id: str
    space_id: str
    status: Literal["cancelling"] = "cancelling"


class ThreadDeleteResponse(BaseModel):
    chat_id: str
    space_id: str
    deleted: bool


class SpaceThreadsDeleteResponse(BaseModel):
    space_id: str
    deleted_threads: int


class TurnRevertRequest(BaseModel):
    entity_ids: list[str] | None = None


class TurnRevertResponse(BaseModel):
    trace_id: str
    space_id: str
    reverted_count: int
    results: list[RevertStepResult] = Field(default_factory=list)


class ThreadCheckpointsResponse(BaseModel):
    chat_id: str
    space_id: str
    checkpoints: list[CheckpointSummary] = Field(default_factory=list)


class ThreadRewindRequest(BaseModel):
    checkpoint_id: str = Field(..., min_length=1)
    revert_trace_ids: list[str] = Field(default_factory=list)


class ThreadRewindResponse(BaseModel):
    chat_id: str
    space_id: str
    target_checkpoint_id: str
    new_checkpoint_id: str
    message_count: int
    reverted_count: int = 0
    revert_results: list[RevertStepResult] = Field(default_factory=list)


class StorePutRequest(BaseModel):
    space_id: str = Field(..., min_length=1)
    namespace: str = Field(default="default", min_length=1)
    key: str = Field(..., min_length=1)
    value: dict[str, Any]


class StoreItemResponse(BaseModel):
    space_id: str
    namespace: str
    key: str
    value: dict[str, Any]
    created_at: str | None = None
    updated_at: str | None = None
    score: float | None = None


class StoreSearchRequest(BaseModel):
    space_id: str = Field(..., min_length=1)
    namespace: str = Field(default="default", min_length=1)
    query: str | None = None
    filter: dict[str, Any] | None = None
    limit: int = Field(default=10, ge=1, le=100)


class StoreSearchResponse(BaseModel):
    items: list[StoreItemResponse] = Field(default_factory=list)


class StoreDeleteResponse(BaseModel):
    space_id: str
    namespace: str
    key: str
    deleted: bool = True


class SpaceStoreDeleteResponse(BaseModel):
    space_id: str
    deleted_items: int


class KnowledgeChunkRequest(BaseModel):
    chunk_index: int = Field(default=0, ge=0)
    content: str = Field(..., min_length=1)
    page: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeUpsertRequest(BaseModel):
    space_id: str = Field(..., min_length=1)
    source_type: str = Field(..., min_length=1)
    source_id: str = Field(..., min_length=1)
    title: str = ""
    url: str | None = None
    content: str = ""
    chunks: list[KnowledgeChunkRequest] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeUpsertResponse(BaseModel):
    space_id: str
    source_type: str
    source_id: str
    chunks_indexed: int


class KnowledgeIngestResponse(BaseModel):
    job_id: str
    status: IngestionJobStatus
    space_id: str
    source_type: str
    source_id: str
    title: str = ""
    pages_parsed: int = 0
    chunks_indexed: int = 0
    error: str = ""


class KnowledgeSearchRequest(BaseModel):
    space_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=50)
    source_types: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class KnowledgeSearchResponse(BaseModel):
    hits: list[KnowledgeHit] = Field(default_factory=list)


class KnowledgeDeleteResponse(BaseModel):
    space_id: str
    deleted: bool = True
    source_type: str | None = None
    source_id: str | None = None


class AgentInfoResponse(BaseModel):
    name: str
    description: str
    capabilities: list[str] = Field(default_factory=list)
    timeout_seconds: float


class AgentListResponse(BaseModel):
    agents: list[AgentInfoResponse] = Field(default_factory=list)


class ModelListResponse(BaseModel):
    models: list[ModelCatalogEntry] = Field(default_factory=list)


class FeedbackCreateRequest(BaseModel):
    space_id: str = Field(..., min_length=1)
    score: float = Field(..., ge=-1.0, le=1.0)
    comment: str = ""
    category: str = ""


class FeedbackListResponse(BaseModel):
    items: list[FeedbackRecord] = Field(default_factory=list)
