"""Sub-agents package — specialized A2A agents."""

from app.agents.sub_agents.task_agent import TaskSubAgent, TASK_AGENT_CARD
from app.agents.sub_agents.project_agent import ProjectSubAgent, PROJECT_AGENT_CARD
from app.agents.sub_agents.cycle_agent import CycleSubAgent, CYCLE_AGENT_CARD
from app.agents.sub_agents.workspace_agent import WorkspaceSubAgent, WORKSPACE_AGENT_CARD

__all__ = [
    "TaskSubAgent", "TASK_AGENT_CARD",
    "ProjectSubAgent", "PROJECT_AGENT_CARD",
    "CycleSubAgent", "CYCLE_AGENT_CARD",
    "WorkspaceSubAgent", "WORKSPACE_AGENT_CARD",
]
