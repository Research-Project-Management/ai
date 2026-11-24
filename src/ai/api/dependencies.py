"""FastAPI dependencies for scoped config injection, auth, rate limit, and SpaceContext."""

import hmac
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import SecretStr

from ai.agents.rag.ingestion.jobs import IngestionJobRunner
from ai.agents.rag.ingestion.pipeline import IngestionPipeline
from ai.agents.rag.retrieval.engine import RetrievalEngine
from ai.api.schemas import ChatRequest, ResumeRequest
from ai.bootstrap import AppContainer
from ai.orchestrator.contracts import AgentRegistry, InputGuard
from ai.shared.config.server import ServerConfig
from ai.shared.contracts.auth import bind_delegation_token
from ai.shared.contracts.context import SpaceContext
from ai.shared.contracts.health import HealthProbe
from ai.shared.contracts.knowledge import KnowledgeStore
from ai.shared.contracts.rate_limit import RateLimiter, RateLimitResult
from ai.shared.contracts.run_control import RunCancellationRegistry
from ai.shared.llm.factory import LLMOverride, bind_llm_override
from ai.shared.resilience.mutations import JournalStore, bind_turn_journal
from ai.shared.resilience.observability import TokenBudget, TokenQuotaStatus, ToolExecutionMetrics
from ai.shared.resilience.tool_runner import bind_tool_metrics


def get_container(request: Request) -> AppContainer:
    """Return the typed AppContainer attached during FastAPI lifespan startup."""
    container: AppContainer = request.app.state.container
    return container


def get_server_config(request: Request) -> ServerConfig:
    """Return ServerConfig from application container."""
    return get_container(request).settings.server


def get_rate_limiter(request: Request) -> RateLimiter:
    """Inject the configured Space-scoped RateLimiter from application container."""
    return get_container(request).rate_limiter


def get_token_budget(request: Request) -> TokenBudget:
    """Inject the configured Space-scoped TokenBudget from application container."""
    return get_container(request).token_budget


def get_health_probes(request: Request) -> tuple[HealthProbe, ...]:
    """Inject registered infrastructure HealthProbes from application container."""
    return get_container(request).health_probes


def get_cancel_registry(request: Request) -> RunCancellationRegistry:
    """Inject active run cancellation registry from application container."""
    return get_container(request).cancel_registry


def get_knowledge_store(request: Request) -> KnowledgeStore:
    """Inject space-isolated KnowledgeStore from application container."""
    return get_container(request).knowledge_store


def get_mutation_journal(request: Request) -> JournalStore:
    """Inject per-turn mutation journal from application container."""
    return get_container(request).mutation_journal


def get_input_guards(request: Request) -> tuple[InputGuard, ...]:
    """Inject inbound message validation guards from application container."""
    return get_container(request).input_guards


def get_agent_registry(request: Request) -> AgentRegistry:
    """Inject specialist AgentRegistry from application container."""
    return get_container(request).agent_registry


def get_checkpointer(request: Request) -> BaseCheckpointSaver:
    """Inject thread checkpointer from application container."""
    return get_container(request).checkpointer


def get_ingestion_pipeline(request: Request) -> IngestionPipeline:
    """Inject write-side RAG IngestionPipeline from application container."""
    return get_container(request).ingestion_pipeline


def get_ingestion_job_runner(request: Request) -> IngestionJobRunner:
    """Inject bounded async IngestionJobRunner from application container."""
    return get_container(request).ingestion_job_runner


def get_retrieval_engine(request: Request) -> RetrievalEngine:
    """Inject read-side RAG RetrievalEngine from application container."""
    return get_container(request).retrieval_engine


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


def extract_bearer_token(request: Request) -> str:
    """Extract the bearer token from the request's `Authorization` header."""
    auth_header = request.headers.get("authorization", "").strip()
    if auth_header[:7].lower() == "bearer ":
        return auth_header[7:].strip()
    return ""


@contextmanager
def bind_run_context(
    *,
    delegation_token: str = "",
    llm_override: LLMOverride | None = None,
    tool_metrics: ToolExecutionMetrics | None = None,
    mutation_journal: JournalStore | None = None,
    space_id: str = "",
    trace_id: str = "",
) -> Iterator[None]:
    """Bind per-run ContextVars (delegation auth, BYOK override, tool metrics, mutation journal)."""
    with (
        bind_delegation_token(delegation_token),
        bind_llm_override(llm_override),
        bind_tool_metrics(tool_metrics),
        bind_turn_journal(mutation_journal, space_id=space_id, trace_id=trace_id),
    ):
        yield


def build_space_context(
    payload: ChatRequest | ResumeRequest,
    raw_request: Request,
    x_trace_id: str | None = None,
) -> SpaceContext:
    """Extract verified Flux space scope and masked delegation token."""
    token = extract_bearer_token(raw_request)
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
