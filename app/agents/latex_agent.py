"""LaTeX Agent — generates LaTeX code for equations, tables, and documents."""

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from app.agents.base import AgentState, BaseAgent
from app.agents.registry import register_agent
from app.config import get_llm


LATEX_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert LaTeX code generator. Generate clean, "
            "compilable LaTeX code based on the user's request.\n\n"
            "Rules:\n"
            "- Always wrap the LaTeX code in ```latex ... ``` blocks\n"
            "- For standalone expressions, provide just the LaTeX math\n"
            "- For full documents, include \\documentclass, preamble, and body\n"
            "- Use common packages: amsmath, amssymb, graphicx, booktabs\n"
            "- Add brief comments explaining complex constructs\n"
            "- If generating a table, use booktabs style (\\toprule, \\midrule, \\bottomrule)\n"
            "- Ensure the output is valid and compilable LaTeX",
        ),
        ("human", "{request}"),
    ]
)


@register_agent
class LaTeXAgent(BaseAgent):
    """
    LaTeX code generation agent.

    Generates LaTeX for: mathematical equations, tables, figures,
    full paper templates, slide decks (beamer), and more.
    """

    name = "latex"
    description = (
        "Generates LaTeX code for mathematical equations, tables, figures, "
        "paper templates, and beamer slides. Use when the user asks to "
        "create, write, or generate LaTeX, formulas, equations, or "
        "scientific document formatting."
    )

    async def run(self, state: AgentState) -> AgentState:
        llm = get_llm()

        messages = state.get("messages", [])
        request = ""
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                request = msg.content
                break
        if not request and messages:
            request = str(messages[-1].content)

        chain = LATEX_PROMPT | llm
        response = await chain.ainvoke({"request": request})
        content = response.content if isinstance(response.content, str) else str(response.content)

        state["agent_output"] = {
            "agent_name": self.name,
            "content": content,
            "sources": [],
            "metadata": {"type": "latex"},
        }
        state["messages"] = [AIMessage(content=content)]
        return state
