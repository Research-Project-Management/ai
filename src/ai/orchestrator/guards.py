"""Built-in input validation and output integrity guardrails."""

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from ai.orchestrator.citations import sanitize_citation_markers
from ai.orchestrator.contracts import CitationItem, DraftOutput, InputGuardViolation


def _strip_null_bytes(content: Any) -> tuple[Any, int]:
    if isinstance(content, str):
        cleaned = content.replace("\x00", "")
        return cleaned, len(cleaned)
    if isinstance(content, list):
        total = 0
        cleaned_blocks: list[Any] = []
        for block in content:
            if isinstance(block, str):
                piece = block.replace("\x00", "")
                total += len(piece)
                cleaned_blocks.append(piece)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                piece = block["text"].replace("\x00", "")
                total += len(piece)
                cleaned_blocks.append({**block, "text": piece})
            else:
                cleaned_blocks.append(block)
        return cleaned_blocks, total
    text = str(content).replace("\x00", "")
    return text, len(text)


class DefaultInputGuard:
    """Reject oversized, empty, or over-length inbound message lists and strip null bytes."""

    def __init__(
        self,
        *,
        max_messages: int = 100,
        max_message_chars: int = 32_000,
    ) -> None:
        self._max_messages = max_messages
        self._max_message_chars = max_message_chars

    async def validate_messages(
        self,
        messages: Sequence[BaseMessage],
        config: RunnableConfig,
    ) -> list[BaseMessage]:
        del config
        if not messages:
            raise InputGuardViolation("At least one message is required")
        if len(messages) > self._max_messages:
            raise InputGuardViolation(
                f"Too many messages: {len(messages)} exceeds limit of {self._max_messages}"
            )

        sanitized: list[BaseMessage] = []
        has_non_blank = False
        for msg in messages:
            cleaned_content, char_len = _strip_null_bytes(msg.content)
            if char_len > self._max_message_chars:
                raise InputGuardViolation(
                    f"Message length {char_len} exceeds limit of {self._max_message_chars}"
                )
            if isinstance(cleaned_content, str):
                if cleaned_content.strip():
                    has_non_blank = True
            elif char_len > 0:
                has_non_blank = True
            sanitized.append(msg.model_copy(update={"content": cleaned_content}))

        if not has_non_blank:
            raise InputGuardViolation("Messages contain no non-whitespace content")
        return sanitized


class CitationIntegrityGuard:
    """Output guardrail deduplicating citations and stripping phantom/out-of-bounds [[cite:N]] markers."""

    supports_live_streaming: bool = True

    async def validate(
        self,
        draft: DraftOutput,
        state: Any,
        config: RunnableConfig,
    ) -> DraftOutput:
        del state, config
        seen: set[tuple[int | None, str, int | None, str]] = set()
        unique: list[CitationItem] = []
        for cite in draft.citations:
            if not cite.source_id.strip():
                continue
            key = (cite.index, cite.source_id, cite.page, cite.snippet)
            if key not in seen:
                seen.add(key)
                unique.append(cite)

        valid_indices = frozenset(
            cite.index if cite.index is not None else idx
            for idx, cite in enumerate(unique, start=1)
        )
        cleaned_text, _, _ = sanitize_citation_markers(
            draft.content,
            valid_indices=valid_indices,
        )
        return draft.model_copy(
            update={
                "content": cleaned_text,
                "citations": tuple(unique),
            }
        )
