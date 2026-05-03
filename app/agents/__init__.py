"""
Agents package — import all agents here to trigger registration.

When this package is imported, every @register_agent decorated class
is instantiated and added to the global registry automatically.
"""

# Import all agents to register them
from app.agents.chat_agent import ChatAgent  # noqa: F401
from app.agents.rag_agent import RAGAgent  # noqa: F401
from app.agents.analyze_agent import AnalyzeAgent  # noqa: F401
from app.agents.latex_agent import LaTeXAgent  # noqa: F401
from app.agents.task_agent import TaskAgent  # noqa: F401
from app.agents.web_search_agent import WebSearchAgent  # noqa: F401
from app.agents.action_agent import ActionAgent  # noqa: F401
from app.agents.latex_editor_agent import LaTeXEditorAgent  # noqa: F401

# Re-export key utilities
from app.agents.registry import get_agent, list_agents, agent_descriptions  # noqa: F401
from app.agents.router import get_graph  # noqa: F401
from app.agents.graph import get_pipeline  # noqa: F401

