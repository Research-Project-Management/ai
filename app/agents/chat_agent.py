"""Chat Agent — general conversational AI."""

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.base import AgentState, BaseAgent
from app.agents.registry import register_agent
from app.config import get_llm


@register_agent
class ChatAgent(BaseAgent):
    """
    General-purpose conversational agent.

    Handles greetings, explanations, Q&A, and anything that doesn't
    require specialised tools (RAG, LaTeX, etc.).
    """

    name = "chat"
    description = (
        "General conversational AI. Handles greetings, explanations, "
        "free-form Q&A, and any request that does not need document "
        "retrieval, data analysis, LaTeX generation, or task planning."
    )

    async def run(self, state: AgentState) -> AgentState:
        llm = get_llm()

        # Build message list from state
        messages = state.get("messages", [])

        response = await llm.ainvoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)

        state["agent_output"] = {
            "agent_name": self.name,
            "content": content,
            "sources": [],
            "metadata": {},
        }
        state["messages"] = [AIMessage(content=content)]
        return state
