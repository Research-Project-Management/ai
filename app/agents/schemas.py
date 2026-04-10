"""Pydantic schemas for structured LLM output in the Supervisor Pipeline.

These models are used with `llm.with_structured_output()` to get
reliable, typed responses from the LLM during routing and judging.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Supervisor's routing decision.

    The supervisor uses this to choose which agent/pipeline handles
    the user's request.  Adding a new agent to the registry
    automatically makes it available as a route option.
    """

    route: str = Field(
        ...,
        description="The agent name or pipeline route to handle the request.",
    )
    reasoning: str = Field(
        "",
        description="Brief reasoning for the routing decision.",
    )


class RagJudge(BaseModel):
    """Judge evaluating whether RAG context is sufficient.

    Used after retrieval to decide if the gathered context can answer
    the question or if web search is needed as fallback.
    """

    sufficient: bool = Field(
        ...,
        description=(
            "True if the retrieved context is sufficient and relevant "
            "to fully answer the question; False otherwise."
        ),
    )
    reasoning: str = Field(
        "",
        description="Brief reasoning for the judgement.",
    )
