"""
Agents package — imports all agents to trigger registration.

Layout:
  base.py              — BaseAgent ABC
  registry.py          — @register_agent decorator + lookup
  card.py              — AgentCard / AgentTask / AgentResult data classes

  Standalone agents (registered in the main router):
    chat_agent.py          — General conversation
    rag_agent.py           — Document Q&A
    analyze_agent.py       — Document analysis
    latex_agent.py         — LaTeX export
    latex_editor_agent.py  — Collaborative LaTeX editing
    task_planner_agent.py  — Task suggestion / planning (AI-only, no CRUD)
    web_search_agent.py    — Web + academic search

  Orchestrator (A2A):
    orchestrator.py        — stream_orchestrator() entry point
    action_agent.py        — route-level action dispatcher

  Workspace sub-agents (invoked via orchestrator tools):
    sub_agents/
      base_sub_agent.py    — shared BaseSubAgent template
      task_agent.py        — Task CRUD via RPM-BE
      project_agent.py     — Project & member management
      cycle_agent.py       — Research cycle management
      workspace_agent.py   — Workspace overview & analytics
"""

# ── Register all standalone agents ────────────────────────────────────────
from app.agents.chat_agent import ChatAgent  # noqa: F401
from app.agents.rag_agent import RAGAgent  # noqa: F401
from app.agents.analyze_agent import AnalyzeAgent  # noqa: F401
from app.agents.latex_agent import LaTeXAgent  # noqa: F401
from app.agents.task_planner_agent import TaskAgent  # noqa: F401  (renamed from task_agent)
from app.agents.web_search_agent import WebSearchAgent  # noqa: F401
from app.agents.action_agent import ActionAgent  # noqa: F401
from app.agents.latex_editor_agent import LaTeXEditorAgent  # noqa: F401

# ── Re-export key utilities ────────────────────────────────────────────────
from app.agents.registry import get_agent, list_agents, agent_descriptions  # noqa: F401
from app.agents.graph import get_pipeline  # noqa: F401
