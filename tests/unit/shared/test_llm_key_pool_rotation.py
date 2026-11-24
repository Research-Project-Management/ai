"""Seam tests for per-request round-robin key rotation in ContextAwareChatRunnable."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from pydantic import SecretStr

from ai.shared.config.llm import LLMConfig
from ai.shared.llm.factory import (
    ContextAwareChatRunnable,
    LLMOverride,
    bind_llm_override,
    build_chat_model,
)


def _extract_primary_key(target: Any) -> str:
    primary = target.runnable if isinstance(target, RunnableWithFallbacks) else target
    key = getattr(primary, "openai_api_key", None)
    assert isinstance(key, SecretStr)
    return key.get_secret_value()


def test_context_aware_runnable_rotates_prebuilt_key_pool_per_request() -> None:
    cfg = LLMConfig(
        api_key=SecretStr("key-1"),
        extra_api_keys=SecretStr("key-2, key-3"),
        agent_api_key=SecretStr(""),
        router_api_key=SecretStr(""),
        model="gpt-4o-mini",
        agent_model="gpt-4o-mini",
        router_model="gpt-4o-mini",
    )
    r0 = build_chat_model(cfg, role="agent", key_index=0)
    proxy = ContextAwareChatRunnable(r0, config=cfg, role="agent")

    # 6 consecutive requests cycle cleanly across key-1 -> key-2 -> key-3 -> key-1 -> key-2 -> key-3
    keys_seen = [_extract_primary_key(proxy._active_target()) for _ in range(6)]
    assert keys_seen == ["key-1", "key-2", "key-3", "key-1", "key-2", "key-3"]


async def test_context_aware_runnable_rotates_platform_pool_on_ainvoke_and_respects_byok() -> None:
    cfg = LLMConfig(
        api_key=SecretStr("key-1"),
        extra_api_keys=SecretStr("key-2, key-3"),
        agent_api_key=SecretStr(""),
        router_api_key=SecretStr(""),
        model="gpt-4o-mini",
        agent_model="gpt-4o-mini",
        router_model="gpt-4o-mini",
    )
    hits: list[str] = []

    def _make_stub(label: str) -> RunnableLambda[LanguageModelInput, AIMessage]:
        def _call(_inp: LanguageModelInput) -> AIMessage:
            hits.append(label)
            return AIMessage(content=f"from-{label}")

        return RunnableLambda(_call)

    pool = [_make_stub("k1"), _make_stub("k2"), _make_stub("k3")]
    proxy = ContextAwareChatRunnable(
        pool[0],
        config=cfg,
        role="agent",
        platform_pool=pool,
    )

    outputs = [await proxy.ainvoke(f"msg-{i}") for i in range(5)]
    assert [m.content for m in outputs] == [
        "from-k1",
        "from-k2",
        "from-k3",
        "from-k1",
        "from-k2",
    ]
    assert hits == ["k1", "k2", "k3", "k1", "k2"]

    # BYOK override still bypasses the platform pool for agent role
    override = LLMOverride(
        api_key=SecretStr("sk-user-byok"),
        model="custom-byok-model",
        base_url="https://api.example.org/v1",
    )
    with bind_llm_override(override):
        target = proxy._active_target()
        assert _extract_primary_key(target) == "sk-user-byok"
