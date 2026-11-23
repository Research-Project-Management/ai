"""FastAPI dependencies for scoped config injection, auth, rate limit, and SpaceContext."""

import hmac
import uuid
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from pydantic import SecretStr

from ai.agents.rag.ingestion.jobs import IngestionJobRunner
from ai.agents.rag.ingestion.pipeline import IngestionPipeline
from ai.agents.rag.retrieval.engine import RetrievalEngine
from ai.api.schemas import ChatRequest, ResumeRequest
from ai.orchestrator.contracts import AgentRegistry, InputGuard
from ai.shared.config.server import ServerConfig
from ai.shared.contracts.context import SpaceContext
from ai.shared.contracts.health import HealthProbe
from ai.shared.contracts.knowledge import KnowledgeStore
from ai.shared.contracts.rate_limit import RateLimiter, RateLimitResult
from ai.shared.contracts.run_control import FeedbackSink, RunCancellationRegistry
from ai.shared.resilience.mutations import TurnMutationJournal
from ai.shared.resilience.observability import TokenBudget, TokenQuotaStatus


def get_server_config(request: Request) -> ServerConfig:
    """Inject only ServerConfig from app state (Interface Segregation)."""
    return request.app.state.settings.server


def get_rate_limiter(request: Request) -> RateLimiter:
    """Inject the configured Space-scoped RateLimiter from app state."""
    return request.app.state.rate_limiter


def get_token_budget(request: Request) -> TokenBudget:
    """Inject the configured Space-scoped TokenBudget from app state."""
    return request.app.state.token_budget


def get_health_probes(request: Request) -> tuple[HealthProbe, ...]:
    """Inject registered infrastructure HealthProbes from app state."""
    return request.app.state.health_probes


def get_cancel_registry(request: Request) -> RunCancellationRegistry:
    """Inject active run cancellation registry from app state."""
    return request.app.state.cancel_registry


def get_memory_store(request: Request) -> BaseStore:
    """Inject cross-thread long-term BaseStore from app state."""
    return request.app.state.memory_store


def get_knowledge_store(request: Request) -> KnowledgeStore:
    """Inject space-isolated KnowledgeStore from app state."""
    return request.app.state.knowledge_store


def get_feedback_sink(request: Request) -> FeedbackSink:
    """Inject run feedback collector from app state."""
    return request.app.state.feedback_sink


def get_mutation_journal(request: Request) -> TurnMutationJournal:
    """Inject per-turn mutation journal from app state."""
    return request.app.state.mutation_journal


def get_input_guards(request: Request) -> tuple[InputGuard, ...]:
    """Inject inbound message validation guards from app state."""
    return request.app.state.input_guards


def get_agent_registry(request: Request) -> AgentRegistry:
    """Inject specialist AgentRegistry from app state."""
    return request.app.state.agent_registry


def get_checkpointer(request: Request) -> BaseCheckpointSaver:
    """Inject thread checkpointer from app state."""
    return request.app.state.checkpointer


def get_ingestion_pipeline(request: Request) -> IngestionPipeline:
    """Inject write-side RAG IngestionPipeline from app state."""
    return request.app.state.ingestion_pipeline


def get_ingestion_job_runner(request: Request) -> IngestionJobRunner:
    """Inject bounded async IngestionJobRunner from app state."""
    return request.app.state.ingestion_job_runner


def get_retrieval_engine(request: Request) -> RetrievalEngine:
    """Inject read-side RAG RetrievalEngine from app state."""
    return request.app.state.retrieval_engine


def verify_internal_key(
    server_cfg: Annotated[ServerConfig, Depends(get_server_config)],
    x_internal_key: Annotated[str | None, Header(alias="X-Internal-Key")] = None,
) -> None:
    """Validate the shared internal API key when configured."""
    expected = server_cfg.internal_api_key.get_secret_value()
    if not expected:
        return
    if not x_internal_key or not hmac.compare_digest(x_internal_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal API key",
        )


def build_space_context(
    payload: ChatRequest | ResumeRequest,
    raw_request: Request,
    x_trace_id: str | None = None,
) -> SpaceContext:
    """Extract verified Flux space scope and masked delegation token."""
    auth_header = raw_request.headers.get("authorization", "")
    token = auth_header[7:].strip() if auth_header[:7].lower() == "bearer " else ""
    trace_id = (x_trace_id or "").strip() or uuid.uuid4().hex[:12]
    raw_chat_id = (
        (payload.chat_id or "").strip()
        if isinstance(payload, ChatRequest)
        else payload.chat_id.strip()
    )
    resolved_chat_id = raw_chat_id or f"stateless-{uuid.uuid4().hex[:12]}"
    return SpaceContext(
        space_id=payload.space_id,
        chat_id=resolved_chat_id,
        page_id=payload.page_id,
        document_ids=tuple(payload.document_ids),
        delegation_token=SecretStr(token),
        trace_id=trace_id,
    )


def rate_limit_headers(result: RateLimitResult) -> dict[str, str]:
    """Format standard rate-limit response headers."""
    return {
        "X-RateLimit-Limit": str(result.limit),
        "X-RateLimit-Remaining": str(result.remaining),
        "X-RateLimit-Reset": str(result.reset_after_seconds),
    }


async def enforce_space_rate_limit(
    space_id: str,
    limiter: RateLimiter,
) -> RateLimitResult:
    """Enforce per-`space_id` sliding-window quota and return 429 when exceeded."""
    result = await limiter.check(space_id)
    if not result.allowed:
        headers = {
            "Retry-After": str(max(1, result.reset_after_seconds)),
            **rate_limit_headers(result),
        }
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "rate_limit_exceeded",
                "space_id": space_id,
                "limit": result.limit,
                "remaining": result.remaining,
                "reset_after_seconds": result.reset_after_seconds,
                "reason": result.reason,
            },
            headers=headers,
        )
    return result


async def enforce_space_token_budget(
    space_id: str,
    budget: TokenBudget,
) -> TokenQuotaStatus:
    """Enforce per-`space_id` token budget and return 429 when quota is exhausted."""
    quota = await budget.check(space_id)
    if not quota.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "token_budget_exceeded",
                "space_id": space_id,
                "used_tokens": quota.used_tokens,
                "limit_tokens": quota.limit_tokens,
                "remaining_tokens": quota.remaining_tokens,
                "estimated_cost_usd": quota.estimated_cost_usd,
            },
        )
    return quota
