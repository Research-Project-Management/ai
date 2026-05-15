"""
LangGraph Supervisor Pipeline — Compatibility Shim.

This file now acts as a thin re-export layer so existing imports
(api/chat.py, agents/__init__.py, registry.py) continue to work
without modification.

All logic has been moved to:
  app/agents/pipeline/
    ├── __init__.py         — build_pipeline() + get_pipeline()
    ├── state.py            — PipelineState TypedDict
    ├── reranker.py         — rerank_chunks() (jina/cohere/local/none)
    └── nodes/
        ├── supervisor.py   — supervisor_node() + agent catalog
        ├── rag_node.py     — rag_node() + RagJudge + formatting
        └── web_node.py     — web_research_node() + Tavily

DO NOT add new business logic here — edit the pipeline modules directly.
"""

# ── Re-exports (backward-compatible) ─────────────────────────────────────

from app.agents.pipeline.state import PipelineState  # noqa: F401
from app.agents.pipeline import get_pipeline, build_pipeline  # noqa: F401
from app.agents.pipeline.nodes.supervisor import (  # noqa: F401
    supervisor_node,
    invalidate_agent_catalog_cache,
    _build_agent_catalog,
)
from app.agents.pipeline.nodes.rag_node import rag_node  # noqa: F401
from app.agents.pipeline.nodes.web_node import web_research_node  # noqa: F401
from app.agents.pipeline.reranker import rerank_chunks as _rerank_chunks  # noqa: F401
