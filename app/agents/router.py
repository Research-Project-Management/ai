"""
LangGraph Router — the core orchestrator.

Builds a StateGraph that:
1. Classifies the user's intent via LLM function-calling
2. Routes to the correct specialized agent
3. Returns the agent's output
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from app.agents.base import AgentState
from app.agents.registry import agent_descriptions, get_agent, list_agents
from app.config import get_llm

logger = logging.getLogger(__name__)


# ── Intent Classification ────────────────────────────────────────

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
    """Use LLM to classify which agent should handle the request."""

    # If intent_hint is provided, skip classification
    metadata = state.get("metadata", {})
    if metadata.get("intent_hint"):
        hint = metadata["intent_hint"]
        available = list_agents()
        if hint in available:
            state["intent"] = hint
            logger.info("Intent forced by hint: %s", hint)
            return state

    llm = get_llm(temperature=0)

    # Exclude "web_search" from the LLM classifier — it is only activated
    # programmatically when the user explicitly enables the Web toggle (web_search_sites).
    agents_desc = "\n".join(
        f"- {a['name']}: {a['description']}"
        for a in agent_descriptions()
        if a["name"] != "web_search"
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

    # Parse agent name from response
    content = response.content if isinstance(response.content, str) else str(response.content)

    try:
        # Try JSON parse
        clean = content.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(clean)
        agent_name = parsed.get("agent", "chat")
    except (json.JSONDecodeError, AttributeError):
        # Fallback: look for agent name in text
        available = list_agents()
        agent_name = "chat"
        for name in available:
            if name in content.lower():
                agent_name = name
                break

    # Validate agent exists
    available = list_agents()
    if agent_name not in available:
        agent_name = "chat"

    # Safety guard: LLM must never pick web_search on its own.
    # web_search is only activated when the user explicitly enables the toggle (web_search_sites metadata).
    if agent_name == "web_search" and not metadata.get("web_search_sites"):
        logger.info("LLM picked web_search but web toggle is OFF — redirecting to chat")
        agent_name = "chat"

    # If document_ids are provided, user explicitly wants RAG — prioritise it.
    if metadata.get("document_ids") and agent_name != "rag":
        logger.info("document_ids set — overriding %s → rag", agent_name)
        agent_name = "rag"

    # Only switch to web_search when the user explicitly enabled it
    # (web_search_sites is only sent by the FE when the Web toggle is ON).
    if metadata.get("web_search_sites") and agent_name not in ("rag", "web_search"):
        logger.info("web_search_sites set — overriding %s → web_search", agent_name)
        agent_name = "web_search"

    # If LLM chose rag, verify the collection actually has documents.
    # If empty, fall back to chat — never auto-escalate to web_search.
    if agent_name == "rag":
        try:
            from app.rag.store import get_qdrant_client
            from app.config import get_settings
            settings = get_settings()
            client = get_qdrant_client()
            count = client.count(collection_name=settings.qdrant_collection, exact=False).count
            if count == 0:
                agent_name = "chat"
                logger.info("RAG collection is empty — redirecting to chat")
        except Exception as e:
            logger.warning("Could not check RAG collection size: %s", e)

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
    """
    Build and compile the LangGraph orchestrator.

    Flow: classify_intent → execute_agent → END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify", classify_intent)
    graph.add_node("execute", execute_agent)

    # Define edges
    graph.set_entry_point("classify")
    graph.add_edge("classify", "execute")
    graph.add_edge("execute", END)

    return graph.compile()


# Compiled graph singleton
_graph = None


def get_graph():
    """Get or create the compiled graph singleton."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
