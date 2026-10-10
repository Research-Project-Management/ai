"""Embedding adapters: OpenAI-compatible remote/local embedder, fallback chain, and hash stub."""

import hashlib
import logging
import math
import re
from collections import OrderedDict
from collections.abc import Sequence

from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from ai.shared.config.embedding import EmbeddingConfig
from ai.shared.config.llm import LLMConfig
from ai.shared.contracts.knowledge import Embedder

logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"\w+")
_QUERY_CACHE_CAPACITY = 256


class DeterministicHashEmbedder:
    """Offline L2-normalized token hash embedder for tests and local dev without API keys."""

    def __init__(self, *, dimensions: int = 1024) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dimensions
        tokens = _TOKEN_RE.findall(text.lower())
        for tok in tokens:
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            raw_int = int.from_bytes(digest, "little")
            idx = (raw_int >> 1) % self._dimensions
            sign = 1.0 if (raw_int & 1) == 0 else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class OpenAICompatibleEmbedder:
    """Async dense embedder backed by an OpenAI-compatible `/embeddings` endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimensions: int,
        base_url: str = "",
        timeout: float = 30.0,
        max_retries: int = 2,
        batch_size: int = 16,
        max_input_chars: int = 0,
        send_dimensions: bool = False,
    ) -> None:
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._max_input_chars = max_input_chars
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._client = OpenAIEmbeddings(
            model=model,
            dimensions=dimensions if send_dimensions else None,
            api_key=SecretStr(api_key or "local"),
            base_url=base_url or None,
            timeout=timeout,
            max_retries=max_retries,
            chunk_size=batch_size,
            check_embedding_ctx_length=False,
        )

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _prepare_text(self, text: str) -> str:
        if self._max_input_chars > 0:
            return text[: self._max_input_chars]
        return text

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared = [self._prepare_text(t) for t in texts]
        return await self._client.aembed_documents(prepared)

    async def embed_query(self, text: str) -> list[float]:
        prepared = self._prepare_text(text)
        cached = self._query_cache.get(prepared)
        if cached is not None:
            self._query_cache.move_to_end(prepared)
            return list(cached)
        vec = await self._client.aembed_query(prepared)
        self._query_cache[prepared] = vec
        if len(self._query_cache) > _QUERY_CACHE_CAPACITY:
            self._query_cache.popitem(last=False)
        return list(vec)


class FallbackEmbedder:
    """Same-dimension primary -> fallback embedder (e.g. local TEI/Ollama -> cloud API)."""

    def __init__(self, primary: Embedder, fallback: Embedder) -> None:
        if primary.dimensions != fallback.dimensions:
            raise ValueError(
                f"Primary ({primary.dimensions}d) and fallback ({fallback.dimensions}d) "
                "embedders must match dimensions"
            )
        self._primary = primary
        self._fallback = fallback

    @property
    def dimensions(self) -> int:
        return self._primary.dimensions

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            return await self._primary.embed_documents(texts)
        except Exception as exc:  # noqa: BLE001 - intentional provider failover boundary
            logger.warning("Primary embedder batch failed (%s); switching to fallback", exc)
            return await self._fallback.embed_documents(texts)

    async def embed_query(self, text: str) -> list[float]:
        try:
            return await self._primary.embed_query(text)
        except Exception as exc:  # noqa: BLE001 - intentional provider failover boundary
            logger.warning("Primary embedder query failed (%s); switching to fallback", exc)
            return await self._fallback.embed_query(text)


def build_embedder(
    embedding_cfg: EmbeddingConfig,
    llm_cfg: LLMConfig,
    *,
    allow_remote: bool = True,
) -> Embedder:
    """Build remote/local OpenAI-compatible embedder, optional fallback, or local hash stub."""
    explicit_base = embedding_cfg.base_url.strip()
    primary_key = embedding_cfg.effective_api_key(llm_cfg)
    if (
        not allow_remote
        or embedding_cfg.provider == "local_hash"
        or (not explicit_base and not primary_key)
    ):
        return DeterministicHashEmbedder(dimensions=embedding_cfg.dimensions)

    effective_base = embedding_cfg.effective_base_url(llm_cfg)
    primary = OpenAICompatibleEmbedder(
        api_key=primary_key or "local",
        model=embedding_cfg.model,
        dimensions=embedding_cfg.dimensions,
        base_url=effective_base,
        timeout=embedding_cfg.timeout_seconds,
        max_retries=embedding_cfg.max_retries,
        batch_size=embedding_cfg.batch_size,
        max_input_chars=embedding_cfg.max_input_chars,
        send_dimensions=embedding_cfg.should_send_dimensions(),
    )

    if embedding_cfg.has_fallback:
        fb_base = embedding_cfg.fallback_base_url.strip() or effective_base
        fb_key = (
            embedding_cfg.fallback_api_key.get_secret_value().strip()
            or ("local" if embedding_cfg.fallback_base_url.strip() else primary_key)
        )
        fb_model = embedding_cfg.effective_fallback_model
        secondary = OpenAICompatibleEmbedder(
            api_key=fb_key,
            model=fb_model,
            dimensions=embedding_cfg.dimensions,
            base_url=fb_base,
            timeout=embedding_cfg.timeout_seconds,
            max_retries=embedding_cfg.fallback_max_retries,
            batch_size=embedding_cfg.batch_size,
            max_input_chars=embedding_cfg.max_input_chars,
            send_dimensions=embedding_cfg.should_send_dimensions(fb_model),
        )
        return FallbackEmbedder(primary=primary, fallback=secondary)

    return primary
