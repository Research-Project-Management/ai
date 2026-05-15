"""
Orchestrator context management.

Holds ContextVars used to pass request-scoped data and event queues
between the streaming engine and the delegate tools without polluting
function signatures or breaking LangChain's tool abstraction.
"""

import asyncio
from contextvars import ContextVar

# Holds context for the current request (user_id, project_id, etc.)
_request_ctx: ContextVar[dict] = ContextVar("_request_ctx", default={})

# Queue for sub-agent → orchestrator event forwarding
_sub_agent_event_queue: ContextVar[asyncio.Queue | None] = ContextVar(
    "_sub_agent_event_queue", default=None
)
