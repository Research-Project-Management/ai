import logging

from app.config import get_llm

logger = logging.getLogger(__name__)

def _extract_token(content) -> str:
    """Safely extract text from a LangChain chunk's content.

    Handles:
    - ``str`` — returned as-is
    - ``list`` of content blocks (Anthropic / Groq structured format):
      e.g. [{'type': 'text', 'text': 'Hello'}, {'type': 'text', 'text': ' world'}]
    - Anything else — str() conversion (should not normally happen)
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                # ignore non-text block types (e.g. tool_use, thinking)
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    # fallback — avoid Python repr of complex objects
    return ""

def _translate_to_english(text: str, context_prompt: str = "Translate the following text to English.") -> str:
    """Translate a query to English via LLM so English-indexed sources return better results."""
    try:
        llm = get_llm(temperature=0)
        response = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        f"{context_prompt} "
                        "Return ONLY the translated query, nothing else. "
                        "If it is already in English, return it unchanged."
                    ),
                },
                {"role": "user", "content": text},
            ]
        )
        translated = response.content.strip()
        logger.info("Translated query: %r → %r", text, translated)
        return translated
    except Exception as exc:
        logger.warning("Query translation failed, using original: %s", exc)
        return text
