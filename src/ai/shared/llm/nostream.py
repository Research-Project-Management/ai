"""Suppress intermediate token streaming for internal LLM calls (planners, routers)."""

from langchain_core.runnables import RunnableConfig
from langgraph.constants import TAG_NOSTREAM


def with_nostream(config: RunnableConfig) -> RunnableConfig:
    """Return a copy of RunnableConfig tagged to suppress intermediate token streaming."""
    tags = list(config.get("tags") or ())
    if TAG_NOSTREAM not in tags:
        tags.append(TAG_NOSTREAM)
    return {**config, "tags": tags}
