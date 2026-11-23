"""Platform-managed LLM Model Catalog whitelist and context selector resolution."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, SecretStr

from ai.shared.config.llm import LLMConfig
from ai.shared.llm.factory import LLMOverride

ModelTier = Literal["auto", "fast", "balanced", "pro"]


class ModelCatalogEntry(BaseModel):
    """Public metadata for a user-selectable model option (never exposes secrets)."""

    model_config = ConfigDict(frozen=True)

    id: str
    label: str
    provider: str
    tier: ModelTier
    description: str
    is_default: bool = False


DEFAULT_CATALOG: tuple[ModelCatalogEntry, ...] = (
    ModelCatalogEntry(
        id="auto",
        label="Auto (Tối ưu)",
        provider="Flux Gateway",
        tier="auto",
        description="Tự động chọn model tối ưu theo từng bước",
        is_default=True,
    ),
    ModelCatalogEntry(
        id="gemini-2.5-flash",
        label="Gemini 2.5 Flash",
        provider="Google",
        tier="fast",
        description="Nhanh, tổng hợp tài liệu dài & RAG mượt",
    ),
    ModelCatalogEntry(
        id="deepseek-v3.2",
        label="DeepSeek V3.2",
        provider="DeepSeek",
        tier="balanced",
        description="Gọi tool chính xác, tiết kiệm chi phí",
    ),
    ModelCatalogEntry(
        id="deepseek-v4-flash",
        label="DeepSeek V4 Flash",
        provider="DeepSeek",
        tier="fast",
        description="Siêu nhanh, gọn nhẹ cho tác vụ đơn giản",
    ),
    ModelCatalogEntry(
        id="claude-sonnet-4-6",
        label="Claude Sonnet 4.6",
        provider="Anthropic",
        tier="pro",
        description="Viết lách & phân tích chuyên sâu",
    ),
)

_SELECTABLE_IDS: frozenset[str] = frozenset(
    entry.id for entry in DEFAULT_CATALOG if entry.id != "auto"
)


def list_model_catalog() -> list[ModelCatalogEntry]:
    """Return ordered list of selectable models exposed to backend/frontend."""
    return list(DEFAULT_CATALOG)


def resolve_catalog_override(
    raw_context: dict[str, Any] | None,
    config: LLMConfig,
) -> LLMOverride | None:
    """Resolve `context['model_id']` against the whitelist into a platform-key `LLMOverride`."""
    if not isinstance(raw_context, dict):
        return None
    model_id = str(raw_context.get("model_id") or "").strip()
    if not model_id or model_id == "auto" or model_id not in _SELECTABLE_IDS:
        return None
    raw_key = config.api_key.get_secret_value().strip()
    if not raw_key:
        return None
    return LLMOverride(
        api_key=SecretStr(raw_key),
        model=model_id,
        base_url=config.base_url.strip(),
    )
