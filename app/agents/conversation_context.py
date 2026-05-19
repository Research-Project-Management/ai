"""Conversation context compaction helpers.

These helpers keep follow-up questions grounded without sending an unbounded
chat transcript to every model call.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


def _role_of(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return "User"
    if isinstance(message, AIMessage):
        return "Assistant"
    if isinstance(message, SystemMessage):
        return "System"
    return getattr(message, "type", "Message").title()


def _content_of(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return str(content)


def _squash(text: str, max_chars: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + " ...[truncated]"


def build_conversation_memory(
    messages: list[BaseMessage],
    *,
    max_recent_messages: int = 8,
    max_message_chars: int = 700,
    max_memory_chars: int = 5000,
) -> str:
    """Return compact context for all messages before the current user request."""
    if not messages:
        return ""

    history = messages[:-1] if isinstance(messages[-1], HumanMessage) else messages[:]
    if not history:
        return ""

    recent = history[-max_recent_messages:]
    older = history[:-max_recent_messages]
    sections: list[str] = []

    if older:
        older_lines = [
            f"{_role_of(msg)}: {_squash(_content_of(msg), 280)}"
            for msg in older[-12:]
            if _content_of(msg).strip()
        ]
        if older_lines:
            sections.append(
                "Earlier conversation, compacted. Preserve user goals, decisions, "
                "entities, constraints, and unresolved follow-ups:\n"
                + "\n".join(older_lines)
            )

    recent_lines = [
        f"{_role_of(msg)}: {_squash(_content_of(msg), max_message_chars)}"
        for msg in recent
        if _content_of(msg).strip()
    ]
    if recent_lines:
        sections.append("Recent conversation turns:\n" + "\n".join(recent_lines))

    memory = "\n\n".join(sections).strip()
    if len(memory) > max_memory_chars:
        memory = memory[-max_memory_chars:]
        memory = "...[earlier compacted context omitted]\n" + memory
    return memory


def compact_messages_for_llm(
    messages: list[BaseMessage],
    *,
    max_recent_messages: int = 10,
    max_message_chars: int = 2200,
    max_memory_chars: int = 6000,
) -> tuple[list[BaseMessage], str]:
    """Keep recent chat verbatim-ish and move older turns into system memory."""
    memory = build_conversation_memory(
        messages,
        max_recent_messages=max_recent_messages,
        max_memory_chars=max_memory_chars,
    )

    if len(messages) <= max_recent_messages:
        recent = messages[:]
    else:
        recent = messages[-max_recent_messages:]

    compacted: list[BaseMessage] = []
    for msg in recent:
        content = _content_of(msg)
        if len(content) > max_message_chars:
            content = _squash(content, max_message_chars)
        if isinstance(msg, HumanMessage):
            compacted.append(HumanMessage(content=content))
        elif isinstance(msg, AIMessage):
            compacted.append(AIMessage(content=content))
        elif isinstance(msg, SystemMessage):
            compacted.append(SystemMessage(content=content))
        else:
            compacted.append(msg)

    return compacted, memory


def memory_system_message(memory: str) -> SystemMessage | None:
    if not memory.strip():
        return None
    return SystemMessage(
        content=(
            "Conversation memory for follow-up handling. Use it to resolve pronouns, "
            "ellipses, references like 'that/it/continue/above', user preferences, "
            "and previously chosen entities. Do not quote it unless useful.\n\n"
            + memory
        )
    )
