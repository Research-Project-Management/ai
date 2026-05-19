"""Tests for Flux AI response prompt quality rules."""

from app.api.chat import _build_system_prompt


def test_default_prompt_discourages_curt_answers():
    prompt = _build_system_prompt("chat", "", "", {})

    assert "thoughtful, friendly, and useful rather than curt" in prompt
    assert "Do not give a one-line answer" in prompt
    assert "under-answer substantive questions" in prompt


def test_rag_prompt_requires_grounding_and_synthesis():
    prompt = _build_system_prompt("rag", "[D1] file=paper.pdf\nTEXT: example", "", {})

    assert "ONLY use information that appears in the Source Passages" in prompt
    assert "Do NOT use general knowledge or invent information" in prompt
    assert "Explain what the sources mean" in prompt


def test_analyze_prompt_includes_depth_requirements():
    prompt = _build_system_prompt("analyze", "sample context", "", {})

    assert "**Implications**" in prompt
    assert "**Limitations / Caveats**" in prompt
    assert "Response Quality Guide" in prompt
