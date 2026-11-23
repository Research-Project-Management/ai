"""LLM output normalization: text/reasoning channel extraction and JSON object parsing."""

import json
import re
from typing import Any, Literal

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_REASONING_BLOCK_TYPES = frozenset({"thinking", "reasoning"})
_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"

StreamChannel = Literal["reasoning", "text"]


def extract_text(content: Any) -> str:
    """Flatten a LangChain message `content` into plain answer text, excluding thinking blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = str(block.get("type") or "").strip().lower()
                if btype in _REASONING_BLOCK_TYPES:
                    continue
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
        return "".join(parts)
    return str(content)


def extract_reasoning(chunk: Any) -> str:
    """Extract structured reasoning/thinking text from a chunk's metadata or content blocks."""
    parts: list[str] = []
    additional = getattr(chunk, "additional_kwargs", None)
    rc_primary = ""
    if isinstance(additional, dict):
        rc = additional.get("reasoning_content")
        if isinstance(rc, str) and rc:
            rc_primary = rc
            parts.append(rc)
        r_alt = additional.get("reasoning")
        if isinstance(r_alt, str) and r_alt and r_alt != rc_primary:
            parts.append(r_alt)

    content = getattr(chunk, "content", None)
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = str(block.get("type") or "").strip().lower()
            if btype not in _REASONING_BLOCK_TYPES:
                continue
            raw_thinking = block.get("thinking")
            raw_text = raw_thinking if isinstance(raw_thinking, str) else block.get("text")
            if isinstance(raw_text, str) and raw_text:
                parts.append(raw_text)

    return "".join(parts)


def _partial_tag_prefix_len(buf: str, tag: str) -> int:
    """Return length of longest non-empty suffix of `buf` that starts `tag` (case-insensitive)."""
    lowered = buf.lower()
    max_check = min(len(lowered), len(tag) - 1)
    for k in range(max_check, 0, -1):
        if lowered.endswith(tag[:k]):
            return k
    return 0


def _append_part(
    events: list[tuple[StreamChannel, str]],
    channel: StreamChannel,
    delta: str,
) -> None:
    if not delta:
        return
    if events and events[-1][0] == channel:
        events[-1] = (channel, events[-1][1] + delta)
    else:
        events.append((channel, delta))


class StreamChannelSplitter:
    """Split a streaming LLM chunk sequence cleanly into `reasoning` and `text` deltas.

    Handles both provider-native reasoning channels (`additional_kwargs["reasoning_content"]`,
    `thinking` / `reasoning` content blocks) and inline `<think>...</think>` tags split across
    arbitrary chunk boundaries.
    """

    def __init__(self) -> None:
        self._in_think = False
        self._buf = ""

    def feed_chunk(self, chunk: Any) -> list[tuple[StreamChannel, str]]:
        """Process one streamed chunk and emit ready `(channel, delta)` pairs."""
        events: list[tuple[StreamChannel, str]] = []
        reasoning_part = extract_reasoning(chunk)
        if reasoning_part:
            _append_part(events, "reasoning", reasoning_part)

        raw_content = chunk if isinstance(chunk, str) else getattr(chunk, "content", None)
        text_part = extract_text(raw_content)
        if not text_part:
            return events

        self._buf += text_part
        while self._buf:
            if not self._in_think:
                idx = self._buf.lower().find(_OPEN_TAG)
                if idx != -1:
                    _append_part(events, "text", self._buf[:idx])
                    self._buf = self._buf[idx + len(_OPEN_TAG) :]
                    self._in_think = True
                    continue
                hold = _partial_tag_prefix_len(self._buf, _OPEN_TAG)
                emit_len = len(self._buf) - hold
                if emit_len > 0:
                    _append_part(events, "text", self._buf[:emit_len])
                    self._buf = self._buf[emit_len:]
                break

            idx = self._buf.lower().find(_CLOSE_TAG)
            if idx != -1:
                _append_part(events, "reasoning", self._buf[:idx])
                self._buf = self._buf[idx + len(_CLOSE_TAG) :]
                self._in_think = False
                continue
            hold = _partial_tag_prefix_len(self._buf, _CLOSE_TAG)
            emit_len = len(self._buf) - hold
            if emit_len > 0:
                _append_part(events, "reasoning", self._buf[:emit_len])
                self._buf = self._buf[emit_len:]
            break

        return events

    def flush(self) -> list[tuple[StreamChannel, str]]:
        """Flush any held partial-tag suffix at the end of an LLM stream."""
        if not self._buf:
            self._in_think = False
            return []
        channel: StreamChannel = "reasoning" if self._in_think else "text"
        tail = self._buf
        self._buf = ""
        self._in_think = False
        return [(channel, tail)]


def strip_reasoning(text: str) -> str:
    """Remove `<think>...</think>` blocks emitted by reasoning models."""
    return _THINK_BLOCK_RE.sub("", text).strip()


def parse_json_object(raw_text: str) -> dict[str, Any] | None:
    """Parse a JSON object from model output, tolerating `<think>` blocks and code fences.

    Returns `None` when the text is not a JSON object, so callers own their fallback.
    """
    cleaned = strip_reasoning(raw_text)
    fence = _CODE_FENCE_RE.match(cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
