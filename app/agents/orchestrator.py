"""
Orchestrator Agent — Compatibility Shim.

This file acts as a thin re-export layer so existing imports
(api/chat.py, etc.) continue to work without modification.

All logic has been moved to:
  app/agents/orchestrator/
    ├── __init__.py    — Package exports
    ├── context.py     — ContextVars for request state and queue
    ├── tools.py       — Delegate LangChain tools
    └── engine.py      — Streaming graph and synthesis

DO NOT add new business logic here.
"""

from app.agents.orchestrator.engine import stream_orchestrator  # noqa: F401
from app.agents.orchestrator.tools import CONFIRM_REQUIRED_TOOLS  # noqa: F401
