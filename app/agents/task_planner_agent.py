"""Task Planner Agent — suggests tasks based on project progress.

NOTE: This is the *planning* agent (registered in the main agent registry).
      For task *execution* (CRUD via RPM-BE), see sub_agents/task_agent.py.
"""

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from app.agents.base import AgentState, BaseAgent
from app.agents.registry import register_agent
from app.config import get_llm


TASK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are FLux AI, and you are a research project planner and task manager. "
            "Based on the current project context, suggest actionable next "
            "steps and tasks.\n\n"
            "Format your response as:\n"
            "## Suggested Tasks\n"
            "For each task provide:\n"
            "- **Title**: Clear, actionable task name\n"
            "- **Priority**: High / Medium / Low\n"
            "- **Estimated Time**: Rough time estimate\n"
            "- **Description**: What needs to be done and why\n"
            "- **Dependencies**: What must be completed first (if any)\n\n"
            "## Milestones\n"
            "Suggest 2-3 milestones if enough context is provided.\n\n"
            "## Project Context\n{context}",
        ),
        ("human", "{request}"),
    ]
)


@register_agent
class TaskAgent(BaseAgent):
    """
    Task planning agent.

    Analyzes current project progress and suggests next tasks,
    milestones, and timelines for research projects.
    """

    name = "task"
    description = (
        "Suggests tasks and milestones based on current project progress. "
        "Creates actionable research plans, estimates timelines, and "
        "identifies dependencies. Use when the user asks to plan tasks, "
        "suggest next steps, create a roadmap, or manage research progress."
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

        # Build project context from metadata
        metadata = state.get("metadata", {})
        context_parts = []

        if metadata.get("project_id"):
            context_parts.append(f"Project ID: {metadata['project_id']}")
        if metadata.get("completed_tasks"):
            context_parts.append(
                "Completed tasks:\n"
                + "\n".join(f"- {t}" for t in metadata["completed_tasks"])
            )
        if metadata.get("current_tasks"):
            context_parts.append(
                "Current tasks:\n"
                + "\n".join(f"- {t}" for t in metadata["current_tasks"])
            )

        context_text = "\n\n".join(context_parts) if context_parts else "(No project context provided.)"

        chain = TASK_PROMPT | llm
        response = await chain.ainvoke(
            {"context": context_text, "request": request}
        )
        content = response.content if isinstance(response.content, str) else str(response.content)

        state["agent_output"] = {
            "agent_name": self.name,
            "content": content,
            "sources": [],
            "metadata": {"type": "task_suggestion"},
        }
        state["messages"] = [AIMessage(content=content)]
        return state
