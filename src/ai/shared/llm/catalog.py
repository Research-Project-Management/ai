"""Platform-managed LLM Model Catalog whitelist and context selector resolution."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, SecretStr

from ai.shared.config.llm import LLMConfig
from ai.shared.llm.factory import LLMOverride

logger = logging.getLogger(__name__)

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
        label="Auto",
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


def _load_active_catalog() -> tuple[ModelCatalogEntry, ...]:
    raw = (os.getenv("LLM_MODEL_CATALOG_JSON") or "").strip()
    if not raw:
        return DEFAULT_CATALOG
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("LLM_MODEL_CATALOG_JSON must be a non-empty JSON array")
        return tuple(ModelCatalogEntry.model_validate(item) for item in parsed)
    except Exception as exc:  # noqa: BLE001 - fail-open to default catalog
        logger.warning("Invalid LLM_MODEL_CATALOG_JSON, falling back to DEFAULT_CATALOG: %s", exc)
        return DEFAULT_CATALOG


def list_model_catalog() -> list[ModelCatalogEntry]:
    """Return ordered list of selectable models exposed to backend/frontend."""
    return list(_load_active_catalog())


def resolve_catalog_override(
    raw_context: dict[str, Any] | None,
    config: LLMConfig,
) -> LLMOverride | None:
    """Resolve `context['model_id']` against the whitelist into a platform-key `LLMOverride`."""
    if not isinstance(raw_context, dict):
        return None
    model_id = str(raw_context.get("model_id") or "").strip()
    selectable_ids = {entry.id for entry in _load_active_catalog() if entry.id != "auto"}
    if not model_id or model_id == "auto" or model_id not in selectable_ids:
        return None
    raw_key = config.api_key.get_secret_value().strip()
    if not raw_key:
        return None
    return LLMOverride(
        api_key=SecretStr(raw_key),
        model=model_id,
        base_url=config.base_url.strip(),
        is_byok=False,
    )
