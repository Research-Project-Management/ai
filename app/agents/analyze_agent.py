"""Analyze Agent — document and data analysis."""

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from app.agents.base import AgentState, BaseAgent
from app.agents.registry import register_agent
from app.config import get_llm


ANALYZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a data analyst and research paper reviewer. "
            "Analyze the provided content and give structured insights.\n"
            "Format your response with:\n"
            "- **Key Findings** (bullet points)\n"
            "- **Methodology Assessment** (if applicable)\n"
            "- **Strengths & Weaknesses**\n"
            "- **Summary Table** (use markdown table if comparing items)\n\n"
            "## Content to Analyze\n{content}",
        ),
        ("human", "{question}"),
    ]
)


@register_agent
class AnalyzeAgent(BaseAgent):
    """
    Analysis agent for documents, data, and research papers.

    Provides structured analysis: comparisons, key finding extraction,
    methodology review, and statistical summaries.
    """

    name = "analyze"
    description = (
        "Analyzes documents, research papers, and data. Compares papers, "
        "extracts key findings, reviews methodology, and produces "
        "structured summaries with tables. Use when the user asks to "
        "analyze, compare, review, or evaluate content."
    )

    async def run(self, state: AgentState) -> AgentState:
        llm = get_llm()

        messages = state.get("messages", [])
        question = ""
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                question = msg.content
                break
        if not question and messages:
            question = str(messages[-1].content)

        # Use context if available, otherwise just the question
        context_chunks = state.get("context", [])
        content_text = ""
        if context_chunks:
            content_text = "\n\n".join(
                c.get("content", "") for c in context_chunks
            )
        else:
            content_text = "(No specific content provided. Analyze based on the question.)"

        chain = ANALYZE_PROMPT | llm
        response = await chain.ainvoke(
            {"content": content_text, "question": question}
        )
        content = response.content if isinstance(response.content, str) else str(response.content)

        state["agent_output"] = {
            "agent_name": self.name,
            "content": content,
            "sources": [],
            "metadata": {"type": "analysis"},
        }
        state["messages"] = [AIMessage(content=content)]
        return state
