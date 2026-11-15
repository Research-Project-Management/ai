"""Bounded context window selection strategies for multi-turn scalability."""

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


def _message_text_len(message: BaseMessage) -> int:
    content = message.content
    if isinstance(content, str):
        return len(content)
    return len(str(content))


def _truncate_message(message: BaseMessage, max_chars: int) -> BaseMessage:
    content = message.content
    text = content if isinstance(content, str) else str(content)
    if len(text) <= max_chars:
        return message
    truncated = text[: max_chars - 14].rstrip() + "...[truncated]"
    if isinstance(message, HumanMessage):
        return HumanMessage(content=truncated)
    if isinstance(message, AIMessage):
        return AIMessage(content=truncated)
    if isinstance(message, SystemMessage):
        return SystemMessage(content=truncated)
    return message.model_copy(update={"content": truncated})


class SlidingWindowContextPolicy:
    """Keep the newest conversation messages within max_messages and max_chars budgets."""

    def __init__(self, *, max_messages: int = 12, max_chars: int = 24_000) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be >= 1")
        if max_chars < 200:
            raise ValueError("max_chars must be >= 200")
        self._max_messages = max_messages
        self._max_chars = max_chars

    def select_messages(self, messages: Sequence[BaseMessage]) -> list[BaseMessage]:
        if not messages:
            return []

        selected: list[BaseMessage] = []
        total_chars = 0

        for idx, msg in enumerate(reversed(messages)):
            if len(selected) >= self._max_messages:
                break
            msg_len = _message_text_len(msg)
            if idx == 0:
                bounded = _truncate_message(msg, self._max_chars)
                selected.append(bounded)
                total_chars += _message_text_len(bounded)
                continue
            if total_chars + msg_len > self._max_chars:
                break
            selected.append(msg)
            total_chars += msg_len

        selected.reverse()
        return selected
