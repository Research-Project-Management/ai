"""Assembled `rag_agent` vertical-slice module."""

from dataclasses import dataclass

import httpx

from ai.agents.rag.ingestion.chunker import MarkdownStructureChunker
from ai.agents.rag.ingestion.contracts import ChunkingStrategy
from ai.agents.rag.ingestion.lazy_indexer import (
    FileContentFetcher,
    HttpFileContentFetcher,
    LazyPdfIndexer,
)
from ai.agents.rag.ingestion.parsers import ParserRegistry, build_default_parser_registry
from ai.agents.rag.ingestion.pipeline import IngestionPipeline
from ai.agents.rag.retrieval.contracts import Reranker
from ai.agents.rag.retrieval.engine import RetrievalEngine
from ai.agents.rag.retrieval.reranker import build_reranker
from ai.agents.rag.workflow.contracts import RagAgentConfig
from ai.agents.rag.workflow.graph import build_rag_agent_spec
from ai.orchestrator.contracts import AgentSpec
from ai.shared.config.rag import IngestionConfig, RerankConfig
from ai.shared.contracts.knowledge import KnowledgeStore
from ai.shared.llm.factory import ChatRunnable


@dataclass(frozen=True)
class RagModule:
    """Assembled RAG dependencies exposed cleanly for Composition Root wiring."""

    parser_registry: ParserRegistry
    chunker: ChunkingStrategy
    ingestion_pipeline: IngestionPipeline
    reranker: Reranker
    retrieval_engine: RetrievalEngine
    agent_spec: AgentSpec
    lazy_indexer: LazyPdfIndexer | None = None


def build_rag_module(
    *,
    llm: ChatRunnable,
    knowledge_store: KnowledgeStore,
    ingestion_config: IngestionConfig | None = None,
    rerank_config: RerankConfig | None = None,
    agent_config: RagAgentConfig | None = None,
    parser_registry: ParserRegistry | None = None,
    chunker: ChunkingStrategy | None = None,
    reranker: Reranker | None = None,
    allow_remote_rerank: bool = True,
    backend_url: str = "",
    file_fetcher: FileContentFetcher | None = None,
    lazy_indexer: LazyPdfIndexer | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> RagModule:
    """Assemble the ingestion pipeline, retrieval engine, and `rag_agent` AgentSpec."""
    ing_cfg = ingestion_config or IngestionConfig()
    rnk_cfg = rerank_config or RerankConfig()

    active_parsers = parser_registry or build_default_parser_registry()
    active_chunker: ChunkingStrategy = chunker or MarkdownStructureChunker(
        chunk_size_tokens=ing_cfg.chunk_size_tokens,
        chunk_overlap_tokens=ing_cfg.chunk_overlap_tokens,
    )
    pipeline = IngestionPipeline(
        parser_registry=active_parsers,
        chunker=active_chunker,
        knowledge_store=knowledge_store,
        max_file_bytes=ing_cfg.max_file_bytes,
    )
    active_reranker = reranker or build_reranker(
        rnk_cfg,
        allow_remote=allow_remote_rerank,
        http_client=http_client,
    )
    retrieval_engine = RetrievalEngine(
        knowledge_store=knowledge_store,
        reranker=active_reranker,
        candidate_pool_size=rnk_cfg.candidate_pool_size,
        default_top_k=rnk_cfg.top_k,
        default_min_score=rnk_cfg.min_score,
    )
    active_lazy_indexer = lazy_indexer
    if active_lazy_indexer is None and (file_fetcher is not None or backend_url.strip()):
        active_fetcher = file_fetcher or HttpFileContentFetcher(
            backend_url=backend_url,
            max_bytes=ing_cfg.max_file_bytes,
            http_client=http_client,
        )
        active_lazy_indexer = LazyPdfIndexer(
            ingestion_pipeline=pipeline,
            knowledge_store=knowledge_store,
            file_fetcher=active_fetcher,
        )
    spec = build_rag_agent_spec(
        llm,
        retrieval_engine,
        config=agent_config,
        lazy_indexer=active_lazy_indexer,
    )
    return RagModule(
        parser_registry=active_parsers,
        chunker=active_chunker,
        ingestion_pipeline=pipeline,
        reranker=active_reranker,
        retrieval_engine=retrieval_engine,
        agent_spec=spec,
        lazy_indexer=active_lazy_indexer,
    )


__all__ = ["RagModule", "build_rag_module"]
