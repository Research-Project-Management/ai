"""
LangGraph Supervisor Pipeline — Graph Assembly.

Assembles the StateGraph from individual node modules.
This file's only job is to wire nodes together — no business logic here.

Graph Flow::

    supervisor ──┬── rag_node ──┬── web_research_node ── END
                 │              └── END
                 ├── web_research_node ── END
                 └── END  (specialized / chat / greeting)

Extensibility:
  - Add a new agent: @register_agent + import in agents/__init__.py
  - The Supervisor auto-discovers it via the registry
  - No changes needed in this file
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from app.agents.pipeline.state import PipelineState
from app.agents.pipeline.nodes.supervisor import supervisor_node
from app.agents.pipeline.nodes.rag_node import rag_node
from app.agents.pipeline.nodes.web_node import web_research_node

logger = logging.getLogger(__name__)


# ── Conditional edge functions ─────────────────────────────────────────────

def _route_after_supervisor(state: PipelineState) -> str:
    """Conditional edge after supervisor: which node to visit next."""
    route = state.get("route", "chat")
    if route == "rag":
        return "rag_node"
    if route == "web_search":
        return "web_research_node"
    # Specialized agents / chat / greeting → exit graph.
    # chat.py handles these by streaming with the appropriate system prompt.
    return END


def _route_after_rag(state: PipelineState) -> str:
    """Conditional edge after RAG: go to web or finish."""
    return "web_research_node" if state.get("route") == "web" else END


# ── Graph builder ─────────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    """Build and compile the Supervisor Pipeline graph."""
    graph = StateGraph(PipelineState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("web_research_node", web_research_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", _route_after_supervisor)
    graph.add_conditional_edges("rag_node", _route_after_rag)
    graph.add_edge("web_research_node", END)

    return graph.compile()


# ── Singleton ─────────────────────────────────────────────────────────────

_pipeline = None


def get_pipeline():
    """Get or create the compiled pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
        logger.info("Supervisor Pipeline compiled")
    return _pipeline
