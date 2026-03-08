"""
LangGraph Router — orchestrator for specialized agents.

Builds a StateGraph that:
1. Classifies the user's intent via LLM (choosing among specialized agents only)
2. Routes to the correct specialized agent
3. Returns the agent's output

Note: RAG and web_search are NOT classified here.
They are handled by the streaming pipeline in app/api/chat.py.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from app.agents.base import AgentState
from app.agents.registry import get_agent, list_agents
from app.config import get_llm

logger = logging.getLogger(__name__)


# ── Intent Classification ────────────────────────────────────────

# Only these agents are offered to the LLM classifier.
# rag and web_search are decided by the streaming pipeline (chat.py), not the LLM.
_CLASSIFIABLE_AGENTS = {"latex", "analyze", "task", "chat"}

CLASSIFY_SYSTEM_PROMPT = """\
You are an intent classifier for a multi-agent AI system.
Given the user's message, decide which specialized agent should handle it.

Available agents:
{agents}

Respond with ONLY a JSON object: {{"agent": "<agent_name>"}}

Rules:
- Choose the BEST matching agent based on the user's intent
- If unsure, default to "chat"
- Only use agent names from the list above
"""


async def classify_intent(state: AgentState) -> AgentState:
    """Use LLM to classify which specialized agent should handle the request."""

    metadata = state.get("metadata", {})
    # Respect explicit intent hint
    if metadata.get("intent_hint"):
        hint = metadata["intent_hint"]
        if hint in list_agents():
            state["intent"] = hint
            logger.info("Intent forced by hint: %s", hint)
            return state

    llm = get_llm(temperature=0)

    # Only present the classifiable (specialized) agents to the LLM.
    # rag / web_search are resolved by the pipeline in chat.py.
    from app.agents.registry import agent_descriptions
    agents_desc = "\n".join(
        f"- {a['name']}: {a['description']}"
        for a in agent_descriptions()
        if a["name"] in _CLASSIFIABLE_AGENTS
    )

    system_prompt = CLASSIFY_SYSTEM_PROMPT.format(agents=agents_desc)

    # Get the last user message
    messages = state.get("messages", [])
    last_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_msg = msg.content
            break
        elif hasattr(msg, "type") and msg.type == "human":
            last_msg = msg.content
            break
    if not last_msg and messages:
        last_msg = str(messages[-1].content)

    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": last_msg},
        ]
    )

    content = response.content if isinstance(response.content, str) else str(response.content)

    try:
        clean = content.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(clean)
        agent_name = parsed.get("agent", "chat")
    except (json.JSONDecodeError, AttributeError):
        agent_name = "chat"
        for name in _CLASSIFIABLE_AGENTS:
            if name in content.lower():
                agent_name = name
                break

    # Validate — fall back to "chat" for anything not in the classifiable set
    if agent_name not in _CLASSIFIABLE_AGENTS:
        agent_name = "chat"

    state["intent"] = agent_name
    logger.info("Classified intent: %s", agent_name)
    return state


# ── Agent Executor Node ──────────────────────────────────────────

async def execute_agent(state: AgentState) -> AgentState:
    """Run the selected agent."""
    intent = state.get("intent", "chat")
    agent = get_agent(intent)
    logger.info("Executing agent: %s", agent.name)
    return await agent.run(state)


# ── Build the Graph ──────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and compile the LangGraph orchestrator (used by /chat/sync)."""
    graph = StateGraph(AgentState)
    graph.add_node("classify", classify_intent)
    graph.add_node("execute", execute_agent)
    graph.set_entry_point("classify")
    graph.add_edge("classify", "execute")
    graph.add_edge("execute", END)
    return graph.compile()


_graph = None


def get_graph():
    """Get or create the compiled graph singleton."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
