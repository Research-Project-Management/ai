"""Specialist agent capability discovery endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from ai.api.dependencies import get_agent_registry, verify_internal_key
from ai.api.schemas import AgentInfoResponse, AgentListResponse
from ai.orchestrator.contracts import AgentRegistry

router = APIRouter(tags=["agents"], dependencies=[Depends(verify_internal_key)])


@router.get("/agents", response_model=AgentListResponse)
async def list_agents(
    registry: Annotated[AgentRegistry, Depends(get_agent_registry)],
) -> AgentListResponse:
    """Return metadata for all specialist agents registered in the orchestrator."""
    return AgentListResponse(
        agents=[
            AgentInfoResponse(
                name=spec.name,
                description=spec.description,
                capabilities=list(spec.capabilities),
                timeout_seconds=spec.timeout_seconds,
            )
            for spec in registry.specs
        ]
    )
