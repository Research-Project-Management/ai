"""Tests for the agent system — registry, routing, and individual agents."""

import pytest

# Ensure agents are registered
import app.agents  # noqa: F401

from app.agents.registry import agent_descriptions, get_agent, list_agents


class TestAgentRegistry:
    """Test agent registration and discovery."""

    def test_all_agents_registered(self):
        """All 6 agents should be registered."""
        agents = list_agents()
        expected = {"chat", "rag", "analyze", "latex", "task", "web_search"}
        assert set(agents.keys()) == expected

    def test_get_agent_by_name(self):
        """Can retrieve specific agents by name."""
        chat = get_agent("chat")
        assert chat.name == "chat"

        rag = get_agent("rag")
        assert rag.name == "rag"

    def test_get_unknown_agent_raises(self):
        """Requesting non-existent agent raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            get_agent("nonexistent")

    def test_agent_descriptions(self):
        """All agents should have name and description."""
        descs = agent_descriptions()
        assert len(descs) == 6
        for d in descs:
            assert "name" in d
            assert "description" in d
            assert len(d["description"]) > 10  # Not empty

    def test_agents_have_run_method(self):
        """All registered agents should have an async run method."""
        for name, agent in list_agents().items():
            assert hasattr(agent, "run"), f"Agent {name} missing run()"
            assert callable(agent.run)

    def test_web_search_agent_has_fetch_methods(self):
        """WebSearchAgent should have fetch_results methods."""
        ws = get_agent("web_search")
        assert hasattr(ws, "fetch_results"), "WebSearchAgent missing fetch_results()"
        assert hasattr(ws, "fetch_results_with_query"), "WebSearchAgent missing fetch_results_with_query()"
        assert callable(ws.fetch_results)
        assert callable(ws.fetch_results_with_query)
