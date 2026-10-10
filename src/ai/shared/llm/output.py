"""LLM output normalization: message text, reasoning-block stripping, and JSON object parsing."""

import json
import re
from typing import Any

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_text(content: Any) -> str:
    """Flatten a LangChain message `content` (str or content blocks) into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


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
