"""Pure LLM Gateway factory with role tiering, Round-Robin Key Pool, BYOK override, and Fallback."""

import itertools
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal
from weakref import WeakKeyDictionary

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, SecretStr

from ai.shared.config.llm import LLMConfig
from ai.shared.contracts.run_control import RunUsage
from ai.shared.resilience.observability import LLMCallMetrics

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
    is_byok: bool = True


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


def _extract_token_counts(message: Any) -> tuple[int, int]:
    """Extract `(input_tokens, output_tokens)` from LangChain `usage_metadata` or `token_usage`."""
    usage = RunUsage.from_message(message)
    if usage.input_tokens or usage.output_tokens:
        return usage.input_tokens, usage.output_tokens
    meta = getattr(message, "response_metadata", None)
    if isinstance(meta, dict):
        token_usage = meta.get("token_usage")
        if isinstance(token_usage, dict):
            return (
                max(0, int(token_usage.get("prompt_tokens") or 0)),
                max(0, int(token_usage.get("completion_tokens") or 0)),
            )
    return 0, 0


class _MeteredProxyRunnable(Runnable[LanguageModelInput, AIMessage]):
    """Shared metered `invoke`/`ainvoke`/`stream`/`astream` proxy for context-aware chat models."""

    def _resolve_target(self) -> ChatRunnable:
        raise NotImplementedError

    def _record_metric(
        self,
        *,
        start: float,
        ok: bool,
        response: Any = None,
        in_tokens: int = 0,
        out_tokens: int = 0,
    ) -> None:
        raise NotImplementedError

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        start = time.perf_counter()
        response: AIMessage | None = None
        ok = False
        try:
            response = self._resolve_target().invoke(input, config=config, **kwargs)
            ok = True
            return response
        finally:
            in_tok, out_tok = _extract_token_counts(response) if response is not None else (0, 0)
            self._record_metric(
                start=start,
                ok=ok,
                response=response,
                in_tokens=in_tok,
                out_tokens=out_tok,
            )

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        start = time.perf_counter()
        response: AIMessage | None = None
        ok = False
        try:
            response = await self._resolve_target().ainvoke(input, config=config, **kwargs)
            ok = True
            return response
        finally:
            in_tok, out_tok = _extract_token_counts(response) if response is not None else (0, 0)
            self._record_metric(
                start=start,
                ok=ok,
                response=response,
                in_tokens=in_tok,
                out_tokens=out_tok,
            )

    def stream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[AIMessage]:
        start = time.perf_counter()
        in_tok = 0
        out_tok = 0
        last_chunk: Any = None
        ok = False
        try:
            for chunk in self._resolve_target().stream(input, config=config, **kwargs):
                last_chunk = chunk
                c_in, c_out = _extract_token_counts(chunk)
                in_tok += c_in
                out_tok += c_out
                yield chunk
            ok = True
        finally:
            self._record_metric(
                start=start,
                ok=ok,
                response=last_chunk,
                in_tokens=in_tok,
                out_tokens=out_tok,
            )

    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessage]:
        start = time.perf_counter()
        in_tok = 0
        out_tok = 0
        last_chunk: Any = None
        ok = False
        try:
            async for chunk in self._resolve_target().astream(input, config=config, **kwargs):
                last_chunk = chunk
                c_in, c_out = _extract_token_counts(chunk)
                in_tok += c_in
                out_tok += c_out
                yield chunk
            ok = True
        finally:
            self._record_metric(
                start=start,
                ok=ok,
                response=last_chunk,
                in_tokens=in_tok,
                out_tokens=out_tok,
            )


class ContextAwareChatRunnable(_MeteredProxyRunnable):
    """Transparent proxy with per-request key-pool rotation and ContextVar BYOK routing."""

    def __init__(
        self,
        platform_runnable: ChatRunnable,
        *,
        config: LLMConfig,
        role: ModelRole,
        model: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
        metrics: LLMCallMetrics | None = None,
        platform_pool: Sequence[ChatRunnable] | None = None,
    ) -> None:
        super().__init__()
        self._platform = platform_runnable
        self._config = config
        self._role: ModelRole = role
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._metrics = metrics
        self._byok_cache: OrderedDict[tuple[str, str, str], ChatRunnable] = OrderedDict()
        self._rr_counter = itertools.count()
        self._platform_pool: tuple[ChatRunnable, ...] = self._init_platform_pool(
            platform_runnable,
            platform_pool,
        )

    def _init_platform_pool(
        self,
        platform_runnable: ChatRunnable,
        explicit_pool: Sequence[ChatRunnable] | None,
    ) -> tuple[ChatRunnable, ...]:
        if explicit_pool:
            return tuple(explicit_pool)
        key_pool = self._config.role_api_key_pool(self._role)
        if len(key_pool) <= 1:
            return (platform_runnable,)
        primary = getattr(platform_runnable, "runnable", platform_runnable)
        primary_key = getattr(primary, "openai_api_key", None)
        if not isinstance(primary_key, SecretStr):
            return (platform_runnable,)
        raw_primary = primary_key.get_secret_value()
        raw_keys = [k.get_secret_value() for k in key_pool]
        if raw_primary not in raw_keys:
            return (platform_runnable,)
        start_idx = raw_keys.index(raw_primary)
        pool: list[ChatRunnable] = [platform_runnable]
        for offset in range(1, len(key_pool)):
            key_idx = (start_idx + offset) % len(key_pool)
            pool.append(
                build_chat_model(
                    self._config,
                    role=self._role,
                    model=self._model,
                    temperature=self._temperature,
                    timeout=self._timeout,
                    key_index=key_idx,
                )
            )
        return tuple(pool)

    @property
    def runnable(self) -> Any:
        return getattr(self._platform, "runnable", self._platform)

    @property
    def fallbacks(self) -> list[Any]:
        return list(getattr(self._platform, "fallbacks", ()))

    def _active_override(self) -> LLMOverride | None:
        return get_llm_override() if self._role != "router" else None

    def _active_target(self) -> ChatRunnable:
        override = self._active_override()
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
        if len(self._platform_pool) == 1:
            return self._platform_pool[0]
        return self._platform_pool[next(self._rr_counter) % len(self._platform_pool)]

    def _resolve_metric_model_name(self, response: Any = None) -> str:
        if response is not None:
            meta = getattr(response, "response_metadata", None)
            if isinstance(meta, dict):
                reported = str(meta.get("model_name") or "").strip()
                if reported:
                    return reported
        override = self._active_override()
        override_model = override.model.strip() if override is not None else ""
        return _resolve_model(self._config, self._role, override_model or self._model)

    def _record_metric(
        self,
        *,
        start: float,
        ok: bool,
        response: Any = None,
        in_tokens: int = 0,
        out_tokens: int = 0,
    ) -> None:
        if self._metrics is None:
            return
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        model_name = self._resolve_metric_model_name(response)
        self._metrics.record_call(
            self._role,
            model_name,
            latency_ms=elapsed_ms,
            ok=ok,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
        )

    def _resolve_target(self) -> ChatRunnable:
        return self._active_target()

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> ChatRunnable:
        """Return a context-aware runnable that binds `tools` on the active target (including fallbacks)."""
        bind_kwargs = dict(kwargs)
        if tool_choice is not None:
            bind_kwargs["tool_choice"] = tool_choice
        frozen_tools = tuple(tools)
        return _BoundContextAwareRunnable(
            self,
            lambda target: _bind_target_tools(target, frozen_tools, **bind_kwargs),
        )


def _bind_target_tools(
    target: Any,
    tools: Sequence[Any],
    **kwargs: Any,
) -> ChatRunnable:
    bind_fn = getattr(target, "bind_tools", None)
    if callable(bind_fn):
        bound: Any = bind_fn(tools, **kwargs)
        return bound
    primary = getattr(target, "runnable", None)
    fallbacks = getattr(target, "fallbacks", None)
    if primary is not None and fallbacks is not None:
        bound_primary = _bind_target_tools(primary, tools, **kwargs)
        bound_fallbacks = [_bind_target_tools(fb, tools, **kwargs) for fb in fallbacks]
        return bound_primary.with_fallbacks(bound_fallbacks)
    raise TypeError(f"{type(target).__name__} does not support bind_tools()")


class _BoundContextAwareRunnable(_MeteredProxyRunnable):
    """Proxy returned by `ContextAwareChatRunnable.bind_tools`."""

    def __init__(
        self,
        parent: ContextAwareChatRunnable,
        binder: Callable[[ChatRunnable], ChatRunnable],
    ) -> None:
        super().__init__()
        self._parent = parent
        self._binder = binder
        self._bound_cache: WeakKeyDictionary[Any, ChatRunnable] = WeakKeyDictionary()

    def _resolve_target(self) -> ChatRunnable:
        target = self._parent._active_target()
        cached = self._bound_cache.get(target)
        if cached is not None:
            return cached
        bound = self._binder(target)
        try:
            self._bound_cache[target] = bound
        except TypeError:
            pass
        return bound

    def _record_metric(
        self,
        *,
        start: float,
        ok: bool,
        response: Any = None,
        in_tokens: int = 0,
        out_tokens: int = 0,
    ) -> None:
        self._parent._record_metric(
            start=start,
            ok=ok,
            response=response,
            in_tokens=in_tokens,
            out_tokens=out_tokens,
        )


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


def _make_chat_openai(
    *,
    model: str,
    api_key: SecretStr,
    base_url: str | None,
    temperature: float,
    timeout: float,
    max_retries: int,
    max_completion_tokens: int | None,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        max_completion_tokens=max_completion_tokens,
        default_headers=_DEFAULT_HEADERS,
    )


def _build_fallback_model(
    config: LLMConfig,
    *,
    default_model: str,
    temperature: float,
    timeout: float,
    max_completion_tokens: int | None,
) -> ChatOpenAI:
    return _make_chat_openai(
        model=config.fallback_model or default_model,
        api_key=config.fallback_api_key,
        base_url=config.fallback_base_url or config.base_url or None,
        temperature=temperature,
        timeout=timeout,
        max_retries=config.fallback_max_retries,
        max_completion_tokens=max_completion_tokens,
    )


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
        use_platform_fallback = config.has_fallback and not override.is_byok
        primary = _make_chat_openai(
            model=byok_model,
            api_key=override.api_key,
            base_url=byok_base_url,
            temperature=active_temp,
            timeout=active_timeout,
            max_retries=0 if use_platform_fallback else config.max_retries,
            max_completion_tokens=active_max_tokens,
        )
        if use_platform_fallback:
            return primary.with_fallbacks(
                [
                    _build_fallback_model(
                        config,
                        default_model=byok_model,
                        temperature=active_temp,
                        timeout=active_timeout,
                        max_completion_tokens=active_max_tokens,
                    )
                ]
            )
        return primary

    active_model = _resolve_model(config, role, model)
    active_base_url = config.effective_base_url(role) or None
    rotation_idx = next(_key_rotation_counter) if key_index is None else key_index
    primary_key, backup_keys = _rotate_key_pool(config.role_api_key_pool(role), rotation_idx)

    primary = _make_chat_openai(
        model=active_model,
        api_key=primary_key,
        base_url=active_base_url,
        temperature=active_temp,
        timeout=active_timeout,
        max_retries=config.max_retries,
        max_completion_tokens=active_max_tokens,
    )

    fallbacks: list[ChatRunnable] = [
        _make_chat_openai(
            model=active_model,
            api_key=extra_key,
            base_url=active_base_url,
            temperature=active_temp,
            timeout=active_timeout,
            max_retries=config.fallback_max_retries,
            max_completion_tokens=active_max_tokens,
        )
        for extra_key in backup_keys
    ]

    if config.has_fallback:
        fallbacks.append(
            _build_fallback_model(
                config,
                default_model=active_model,
                temperature=active_temp,
                timeout=active_timeout,
                max_completion_tokens=active_max_tokens,
            )
        )

    if fallbacks:
        return primary.with_fallbacks(fallbacks)

    return primary
