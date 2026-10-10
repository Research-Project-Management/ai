"""Pure LLM Gateway factory with role tiering, Round-Robin Key Pool, BYOK override, and Fallback."""

import itertools
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, SecretStr

from ai.shared.config.llm import LLMConfig

ChatRunnable = Runnable[LanguageModelInput, AIMessage]
ModelRole = Literal["default", "router", "agent"]

_DEFAULT_HEADERS = {"User-Agent": "AI-Service/0.1"}
_BYOK_CLIENT_CACHE_CAPACITY = 64
_key_rotation_counter = itertools.count()


class LLMOverride(BaseModel):
    """Per-request BYOK (Bring Your Own Key) and custom model override descriptor."""

    model_config = ConfigDict(frozen=True)

    api_key: SecretStr
    model: str = ""
    base_url: str = ""


_ACTIVE_LLM_OVERRIDE: ContextVar[LLMOverride | None] = ContextVar(
    "flux_active_llm_override",
    default=None,
)


def get_llm_override() -> LLMOverride | None:
    """Return the active request's BYOK LLMOverride (None if using platform keys)."""
    return _ACTIVE_LLM_OVERRIDE.get()


def parse_llm_override(raw_context: dict[str, Any] | None) -> LLMOverride | None:
    """Extract a validated LLMOverride from `context['llm_override']` if present and non-empty."""
    if not isinstance(raw_context, dict):
        return None
    raw = raw_context.get("llm_override")
    if not isinstance(raw, dict):
        return None
    api_key = str(raw.get("api_key") or "").strip()
    if not api_key:
        return None
    model = str(raw.get("model") or "").strip()
    base_url = str(raw.get("base_url") or "").strip()
    if base_url and not base_url.startswith(("http://", "https://")):
        base_url = ""
    return LLMOverride(
        api_key=SecretStr(api_key),
        model=model,
        base_url=base_url,
    )


@contextmanager
def bind_llm_override(override: LLMOverride | None) -> Iterator[None]:
    """Bind a per-request BYOK LLMOverride into the active async ContextVar."""
    token = _ACTIVE_LLM_OVERRIDE.set(override)
    try:
        yield
    finally:
        _ACTIVE_LLM_OVERRIDE.reset(token)


class ContextAwareChatRunnable(Runnable[LanguageModelInput, AIMessage]):
    """Transparent proxy that routes `agent` and `default` calls to BYOK when bound in ContextVar."""

    def __init__(
        self,
        platform_runnable: ChatRunnable,
        *,
        config: LLMConfig,
        role: ModelRole,
        model: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__()
        self._platform = platform_runnable
        self._config = config
        self._role: ModelRole = role
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._byok_cache: OrderedDict[tuple[str, str, str], ChatRunnable] = OrderedDict()

    @property
    def runnable(self) -> Any:
        return getattr(self._platform, "runnable", self._platform)

    @property
    def fallbacks(self) -> list[Any]:
        return list(getattr(self._platform, "fallbacks", ()))

    def _active_target(self) -> ChatRunnable:
        override = get_llm_override() if self._role != "router" else None
        if override is not None:
            raw_key = override.api_key.get_secret_value().strip()
            if raw_key:
                cache_key = (raw_key, override.model.strip(), override.base_url.strip())
                cached = self._byok_cache.get(cache_key)
                if cached is not None:
                    self._byok_cache.move_to_end(cache_key)
                    return cached
                built = build_chat_model(
                    self._config,
                    role=self._role,
                    model=self._model,
                    temperature=self._temperature,
                    timeout=self._timeout,
                    override=override,
                )
                self._byok_cache[cache_key] = built
                if len(self._byok_cache) > _BYOK_CLIENT_CACHE_CAPACITY:
                    self._byok_cache.popitem(last=False)
                return built
        return self._platform

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        return self._active_target().invoke(input, config=config, **kwargs)

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        return await self._active_target().ainvoke(input, config=config, **kwargs)

    def stream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[AIMessage]:
        yield from self._active_target().stream(input, config=config, **kwargs)

    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessage]:
        async for chunk in self._active_target().astream(input, config=config, **kwargs):
            yield chunk


def _resolve_model(config: LLMConfig, role: ModelRole, model: str | None) -> str:
    if model is not None and model.strip():
        return model.strip()
    if role == "router":
        return config.effective_router_model
    if role == "agent":
        return config.effective_agent_model
    return config.model


def _resolve_temperature(config: LLMConfig, role: ModelRole, temperature: float | None) -> float:
    if temperature is not None:
        return temperature
    if role == "router":
        return 0.0
    return config.temperature


def _rotate_key_pool(
    pool: tuple[SecretStr, ...],
    start_index: int,
) -> tuple[SecretStr, tuple[SecretStr, ...]]:
    """Select the primary key at `start_index % len(pool)` and return remaining keys as backups."""
    if not pool:
        return SecretStr("not-set"), ()
    if len(pool) == 1:
        return pool[0], ()
    offset = start_index % len(pool)
    rotated = pool[offset:] + pool[:offset]
    return rotated[0], rotated[1:]


def build_chat_model(
    config: LLMConfig,
    *,
    role: ModelRole = "default",
    model: str | None = None,
    temperature: float | None = None,
    timeout: float | None = None,
    key_index: int | None = None,
    override: LLMOverride | None = None,
) -> ChatRunnable:
    """Build an OpenAI-compatible chat model with Key Pool rotation, BYOK override, and fallbacks."""
    active_temp = _resolve_temperature(config, role, temperature)
    active_timeout = config.timeout_seconds if timeout is None else timeout
    active_max_tokens: int | None = (
        (min(config.max_tokens, 400) if role == "router" else config.max_tokens)
        if config.max_tokens > 0
        else None
    )

    if override is not None and override.api_key.get_secret_value().strip():
        byok_model = _resolve_model(config, role, override.model or model)
        byok_base_url = override.base_url.strip() or config.base_url or None
        return ChatOpenAI(
            model=byok_model,
            api_key=override.api_key,
            base_url=byok_base_url,
            temperature=active_temp,
            timeout=active_timeout,
            max_retries=config.max_retries,
            max_completion_tokens=active_max_tokens,
            default_headers=_DEFAULT_HEADERS,
        )

    active_model = _resolve_model(config, role, model)
    rotation_idx = next(_key_rotation_counter) if key_index is None else key_index
    primary_key, backup_keys = _rotate_key_pool(config.api_key_pool, rotation_idx)

    primary = ChatOpenAI(
        model=active_model,
        api_key=primary_key,
        base_url=config.base_url or None,
        temperature=active_temp,
        timeout=active_timeout,
        max_retries=config.max_retries,
        max_completion_tokens=active_max_tokens,
        default_headers=_DEFAULT_HEADERS,
    )

    fallbacks: list[ChatRunnable] = [
        ChatOpenAI(
            model=active_model,
            api_key=extra_key,
            base_url=config.base_url or None,
            temperature=active_temp,
            timeout=active_timeout,
            max_retries=config.fallback_max_retries,
            max_completion_tokens=active_max_tokens,
            default_headers=_DEFAULT_HEADERS,
        )
        for extra_key in backup_keys
    ]

    if config.has_fallback:
        fallbacks.append(
            ChatOpenAI(
                model=config.fallback_model or active_model,
                api_key=config.fallback_api_key,
                base_url=config.fallback_base_url or config.base_url or None,
                temperature=active_temp,
                timeout=active_timeout,
                max_retries=config.fallback_max_retries,
                max_completion_tokens=active_max_tokens,
                default_headers=_DEFAULT_HEADERS,
            )
        )

    if fallbacks:
        return primary.with_fallbacks(fallbacks)

    return primary
