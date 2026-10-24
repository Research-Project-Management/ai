"""Comprehensive tests for `rag_agent` (`ingestion`, `retrieval`, `workflow`) and wiring."""

import io
from pathlib import Path

import httpx
import pymupdf
import pytest
from docx import Document
from fastapi.testclient import TestClient
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr
from qdrant_client import AsyncQdrantClient

from ai.agents.rag.ingestion.chunker import MarkdownStructureChunker
from ai.agents.rag.ingestion.contracts import UnsupportedDocumentFormatError
from ai.agents.rag.ingestion.parsers import MarkdownTextParser, build_default_parser_registry
from ai.agents.rag.ingestion.pipeline import IngestionPipeline
from ai.agents.rag.retrieval.contracts import RankedPassage, RetrievalQuery
from ai.agents.rag.retrieval.engine import RetrievalEngine
from ai.agents.rag.retrieval.repacker import filter_and_repack
from ai.agents.rag.retrieval.reranker import HttpCrossEncoderReranker, LexicalHybridReranker
from ai.agents.rag.workflow.contracts import ABSTENTION_RESPONSE, RagAgentConfig
from ai.agents.rag.workflow.graph import build_rag_subgraph
from ai.main import create_app
from ai.shared.config.database import DatabaseConfig, PostgresConfig, QdrantConfig, RedisConfig
from ai.shared.config.embedding import EmbeddingConfig
from ai.shared.config.llm import LLMConfig
from ai.shared.config.rate_limit import RateLimitConfig
from ai.shared.config.server import AppEnvironment, ServerConfig
from ai.shared.config.settings import Settings
from ai.shared.contracts.knowledge import KnowledgeDocumentInput, KnowledgeHit
from ai.shared.llm.embeddings import DeterministicHashEmbedder
from ai.shared.llm.factory import ChatRunnable
from ai.shared.persistence.knowledge import QdrantKnowledgeStore


def _test_settings() -> Settings:
    return Settings(
        server=ServerConfig(
            app_env=AppEnvironment.TEST,
            internal_api_key=SecretStr("test-key"),
        ),
        llm=LLMConfig(api_key=SecretStr("test-llm-key")),
        embedding=EmbeddingConfig(dimensions=64, collection_name="test_rag_col"),
        database=DatabaseConfig(
            postgres=PostgresConfig(database_url=SecretStr("")),
            redis=RedisConfig(redis_url=SecretStr("")),
            qdrant=QdrantConfig(qdrant_url=""),
        ),
        rate_limit=RateLimitConfig(enabled=True, requests_per_window=50),
    )


def _make_pdf_bytes(*, title: str, pages_text: list[str]) -> bytes:
    doc = pymupdf.open()
    doc.set_metadata({"title": title})
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    raw = doc.tobytes()
    doc.close()
    return raw


def _make_docx_bytes(*, title: str, section: str, body: str) -> bytes:
    doc = Document()
    doc.core_properties.title = title
    doc.add_heading(section, level=1)
    doc.add_paragraph(body)
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Metric"
    table.cell(0, 1).text = "Score"
    table.cell(1, 0).text = "Recall@10"
    table.cell(1, 1).text = "0.94"
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_markdown_pdf_and_docx_parsers_and_structure_chunker() -> None:
    registry = build_default_parser_registry()

    pdf_bytes = _make_pdf_bytes(
        title="Sparse Transformers Study",
        pages_text=[
            "# Architecture\nSparse routing cuts attention latency by 42 percent.",
            "# Experiments\nBenchmark on LongBench achieved 88.5 accuracy.",
        ],
    )
    pdf_doc = registry.resolve(filename="paper.pdf").parse_bytes(pdf_bytes, filename="paper.pdf")
    assert pdf_doc.title == "Sparse Transformers Study"
    assert len(pdf_doc.pages) == 2
    assert [p.page_number for p in pdf_doc.pages] == [1, 2]

    docx_bytes = _make_docx_bytes(
        title="Lab Protocol",
        section="Results",
        body="Hybrid search outperformed dense baseline across datasets.",
    )
    docx_doc = registry.resolve(filename="notes.docx").parse_bytes(docx_bytes, filename="notes.docx")
    assert docx_doc.title == "Lab Protocol"
    assert "# Results" in docx_doc.pages[0].markdown
    assert "| Metric | Score |" in docx_doc.pages[0].markdown
    assert "| Recall@10 | 0.94 |" in docx_doc.pages[0].markdown

    md_bytes = (
        b"# Overview\n\nIntro summary paragraph.\n\n"
        b"## Table Section\n\n"
        b"| System | F1 |\n| --- | --- |\n| Baseline | 0.71 |\n| Proposed | 0.89 |\n\n"
        b"```python\nprint('atomic code block')\n```"
    )
    md_doc = registry.resolve(mime_type="text/markdown").parse_bytes(md_bytes, filename="guide.md")
    chunker = MarkdownStructureChunker(chunk_size_tokens=80, chunk_overlap_tokens=12)
    chunks = chunker.chunk(md_doc)
    assert len(chunks) >= 2
    assert all(c.content.startswith("[Document: Overview") for c in chunks)
    table_chunks = [c for c in chunks if "| System | F1 |" in c.content]
    assert len(table_chunks) == 1
    assert "| Proposed | 0.89 |" in table_chunks[0].content
    assert "Section: Overview > Table Section" in table_chunks[0].content

    with pytest.raises(UnsupportedDocumentFormatError):
        registry.resolve(filename="archive.zip", mime_type="application/zip")

    parsed_md = MarkdownTextParser().parse_bytes(b"Plain text note", filename="plain.txt")
    assert parsed_md.title == "plain"


async def test_retrieval_engine_reranker_fallback_and_reverse_repacking() -> None:
    def _failing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "reranker unavailable"}, request=request)

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(_failing_handler))
    reranker = HttpCrossEncoderReranker(
        base_url="http://rerank.invalid",
        api_key="rk-secret",
        fallback=LexicalHybridReranker(),
        http_client=async_client,
    )
    candidates = [
        KnowledgeHit(
            source_id="d1",
            source_type="file",
            title="Misc Notes",
            content="General background paragraph on unrelated topics.",
            score=0.95,
            chunk_index=0,
        ),
        KnowledgeHit(
            source_id="d2",
            source_type="file",
            title="Hybrid Retrieval",
            content="Hybrid sparse dense RRF with reverse repacking improves accuracy.",
            score=0.80,
            chunk_index=0,
        ),
    ]
    ranked = await reranker.rerank("reverse repacking hybrid", candidates, top_k=2)
    await async_client.aclose()
    assert len(ranked) == 2
    assert ranked[0].source_id == "d2"

    passages_desc = [
        RankedPassage(
            source_id="a",
            source_type="file",
            content="top score",
            rerank_score=0.92,
            retrieval_score=0.8,
        ),
        RankedPassage(
            source_id="b",
            source_type="file",
            content="mid score",
            rerank_score=0.65,
            retrieval_score=0.7,
        ),
        RankedPassage(
            source_id="c",
            source_type="file",
            content="dropped below threshold",
            rerank_score=0.05,
            retrieval_score=0.3,
        ),
    ]
    repacked, desc = filter_and_repack(passages_desc, min_score=0.20, top_k=5)
    assert [p.source_id for p in desc] == ["a", "b"]
    assert [p.source_id for p in repacked] == ["b", "a"]

    qdrant = AsyncQdrantClient(location=":memory:")
    store = QdrantKnowledgeStore(qdrant, DeterministicHashEmbedder(dimensions=64))
    await store.upsert(
        KnowledgeDocumentInput(
            space_id="s-iso-1",
            source_type="file",
            source_id="doc-alpha",
            title="Alpha Doc",
            content="[Document: Alpha Doc]\nReverse repacking puts highest evidence closest to query.",
        )
    )
    await store.upsert(
        KnowledgeDocumentInput(
            space_id="s-iso-2",
            source_type="file",
            source_id="doc-beta",
            title="Beta Secret",
            content="[Document: Beta Secret]\nSecret reverse repacking benchmark.",
        )
    )
    engine = RetrievalEngine(
        knowledge_store=store,
        reranker=LexicalHybridReranker(),
        default_min_score=0.0,
    )
    batch = await engine.retrieve(
        RetrievalQuery(space_id="s-iso-1", primary_query="reverse repacking")
    )
    await qdrant.close()
    assert len(batch.ranked_desc) == 1
    assert batch.ranked_desc[0].source_id == "doc-alpha"


async def test_rag_agent_crag_retry_and_abstention_on_subgraph() -> None:
    qdrant = AsyncQdrantClient(location=":memory:")
    store = QdrantKnowledgeStore(qdrant, DeterministicHashEmbedder(dimensions=64))
    pipeline = IngestionPipeline(
        parser_registry=build_default_parser_registry(),
        chunker=MarkdownStructureChunker(chunk_size_tokens=200, chunk_overlap_tokens=20),
        knowledge_store=store,
    )
    pdf_bytes = _make_pdf_bytes(
        title="Photon Quantum Memory",
        pages_text=[
            "# Methods\nOur cryogenic resonator achieves 97.2 percent fidelity at 4 Kelvin.",
        ],
    )
    from ai.agents.rag.ingestion.contracts import IngestDocumentRequest

    ingest_res = await pipeline.ingest(
        IngestDocumentRequest(
            space_id="s-crag",
            source_type="file",
            source_id="photon-1",
            filename="photon.pdf",
            raw_bytes=pdf_bytes,
        )
    )
    assert ingest_res.chunks_indexed >= 1

    # Inject a strict min_score engine on attempt 1? Instead, let the query miss on attempt 1
    # (unknown term with high min_score lexical threshold) and hit on attempt 2 rewrite!
    engine = RetrievalEngine(
        knowledge_store=store,
        reranker=LexicalHybridReranker(),
        default_min_score=0.68,
    )

    call_prompts: list[str] = []

    def _crag_llm(inp: LanguageModelInput) -> AIMessage:
        msgs: list[BaseMessage] = list(inp)  # type: ignore[arg-type]
        sys_text = str(msgs[0].content)
        user_text = str(msgs[-1].content)
        call_prompts.append(sys_text + "\n" + user_text)

        if "Return strict JSON only" in sys_text:
            if "insufficient evidence" in sys_text:
                return AIMessage(
                    content=(
                        '{"standalone_query": "cryogenic resonator fidelity 4 Kelvin", '
                        '"search_queries": ["cryogenic resonator fidelity 4 Kelvin"]}'
                    )
                )
            return AIMessage(
                content=(
                    '{"standalone_query": "zzxxyy_nonexistent_token", '
                    '"search_queries": ["zzxxyy_nonexistent_token"]}'
                )
            )
        return AIMessage(content="The cryogenic resonator reaches 97.2% fidelity at 4K [1].")

    subgraph = build_rag_subgraph(
        RunnableLambda(_crag_llm),
        engine,
        config=RagAgentConfig(max_attempts=2, min_passages=1),
    )

    retry_state = await subgraph.ainvoke(
        {
            "query": "What is the fidelity?",
            "space_id": "s-crag",
            "attempt": 0,
        }
    )
    assert retry_state["attempt"] == 2
    assert retry_state["abstain"] is False
    assert "97.2%" in retry_state["final_answer"]
    assert len(retry_state["citations"]) == 1
    assert retry_state["citations"][0].source_id == "photon-1"
    assert retry_state["citations"][0].page == 1
    # Verify reverse-repacked evidence block placed passage before question
    assert "Question: cryogenic resonator fidelity 4 Kelvin" in call_prompts[-1]

    empty_state = await subgraph.ainvoke(
        {
            "query": "What is the fidelity?",
            "space_id": "s-empty",
            "attempt": 0,
        }
    )
    await qdrant.close()
    assert empty_state["abstain"] is True
    assert empty_state["final_answer"] == ABSTENTION_RESPONSE
    assert empty_state["citations"] == []


def test_rag_api_ingest_search_agent_discovery_and_chat_routing() -> None:
    def _chat_and_rag_stub(inp: LanguageModelInput) -> AIMessage:
        msgs: list[BaseMessage] = list(inp)  # type: ignore[arg-type]
        sys_text = str(msgs[0].content)
        if "Return strict JSON only" in sys_text:
            return AIMessage(
                content=(
                    '{"standalone_query": "attention latency 42 percent", '
                    '"search_queries": ["attention latency 42 percent"]}'
                )
            )
        return AIMessage(content="According to the paper, sparse routing reduces latency by 42% [1].")

    stub: ChatRunnable = RunnableLambda(_chat_and_rag_stub)
    app = create_app(settings=_test_settings(), llm=stub, router_llm=stub)
    headers = {"X-Internal-Key": "test-key"}

    pdf_bytes = _make_pdf_bytes(
        title="Attention Paper",
        pages_text=[
            "# Method\nSparse routing cuts attention latency by 42 percent.",
            "# Appendix\nExtra unrelated note about optimizer beta values.",
        ],
    )

    with TestClient(app) as client:
        agents_res = client.get("/agents", headers=headers)
        assert agents_res.status_code == 200
        agent_names = [item["name"] for item in agents_res.json()["agents"]]
        assert "rag_agent" in agent_names

        ingest_res = client.post(
            "/knowledge/ingest",
            headers=headers,
            data={
                "space_id": "s-api",
                "source_type": "file",
                "source_id": "att-1",
                "metadata_json": '{"origin": "upload"}',
            },
            files={"file": ("attention.pdf", pdf_bytes, "application/pdf")},
        )
        assert ingest_res.status_code == 200
        ingest_body = ingest_res.json()
        assert ingest_body["title"] == "Attention Paper"
        assert ingest_body["pages_parsed"] == 2
        assert ingest_body["chunks_indexed"] >= 2

        search_res = client.post(
            "/knowledge/search",
            headers=headers,
            json={"space_id": "s-api", "query": "attention latency 42 percent", "limit": 3},
        )
        assert search_res.status_code == 200
        hits = search_res.json()["hits"]
        assert len(hits) >= 1
        assert hits[0]["source_id"] == "att-1"
        assert hits[0]["page"] == 1

        chat_res = client.post(
            "/chat/sync",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "How much does sparse routing cut latency?"}],
                "space_id": "s-api",
                "chat_id": "rag-chat-1",
                "document_ids": ["att-1"],
                "intent_hint": "rag_agent",
            },
        )
        assert chat_res.status_code == 200
        chat_body = chat_res.json()
        assert chat_body["route"] == "rag_agent"
        assert "42%" in chat_body["content"]
        assert len(chat_body["citations"]) == 1
        assert chat_body["citations"][0]["source_id"] == "att-1"
        assert chat_body["citations"][0]["page"] == 1


def test_single_root_init_convention_for_rag_slice() -> None:
    rag_root = Path("src/ai/agents/rag")
    assert (rag_root / "module.py").is_file()
    assert list(rag_root.rglob("__init__.py")) == []
    assert [p.as_posix() for p in Path("src/ai").rglob("__init__.py")] == [
        "src/ai/__init__.py"
    ]


async def test_qdrant_alias_routing_and_blue_green_switch() -> None:
    qdrant = AsyncQdrantClient(location=":memory:")
    try:
        embedder = DeterministicHashEmbedder(dimensions=64)
        blue = QdrantKnowledgeStore(
            qdrant,
            embedder,
            collection_name="kb_v1",
            collection_alias="kb_active",
        )
        await blue.upsert(
            KnowledgeDocumentInput(
                space_id="space-alias",
                source_type="file",
                source_id="doc-v1",
                title="Blue Paper",
                content="Transformer attention routing baseline v1.",
            )
        )
        assert await qdrant.collection_exists("kb_v1")
        hits_v1 = await blue.search(space_id="space-alias", query="attention routing v1", limit=2)
        assert len(hits_v1) == 1
        assert hits_v1[0].source_id == "doc-v1"

        green = QdrantKnowledgeStore(
            qdrant,
            embedder,
            collection_name="kb_v2",
            collection_alias="kb_active",
        )
        await green.upsert(
            KnowledgeDocumentInput(
                space_id="space-alias",
                source_type="file",
                source_id="doc-v2",
                title="Green Paper",
                content="Hybrid RRF reranked scientific evidence v2.",
            )
        )
        aliases = await qdrant.get_aliases()
        active_alias = next(a for a in aliases.aliases if a.alias_name == "kb_active")
        assert active_alias.collection_name == "kb_v2"

        hits_v2 = await green.search(space_id="space-alias", query="scientific evidence v2", limit=2)
        assert len(hits_v2) == 1
        assert hits_v2[0].source_id == "doc-v2"
    finally:
        await qdrant.close()


def test_ranked_passage_to_citation_cleans_markdown_tables_and_html() -> None:
    raw_table = (
        "|method|Col2|Col3|error (%)|\n"
        "|---|---|---|---|\n"
        "|Maxout [9]<br>NIN [25]<br>DSN [24]|Maxout [9]<br>NIN [25]<br>DSN [24]|Maxout [9]<br>NIN [25]<br>DSN [24]|9.38<br>8.81<br>8.22|\n"
        "||# layers|# params||\n"
        "|FitNet [34]<br>Highway [41, 42]<br>ResNet-56|19<br>19<br>56|2.5M<br>2.3M<br"
    )
    passage = RankedPassage(
        source_id="file-resnet",
        source_type="file",
        title="He_Deep_Residual_Learning_CVPR_2016_paper.pdf",
        content=raw_table,
        rerank_score=0.91,
        retrieval_score=0.85,
        page=6,
    )
    citation = passage.to_citation(max_snippet_chars=280)
    assert citation.page == 6
    assert "<br" not in citation.snippet.lower()
    assert "|---|" not in citation.snippet
    assert "Col2" not in citation.snippet
    assert citation.snippet.count("Maxout [9]") == 1
    assert "Maxout [9]; NIN [25]; DSN [24] · 9.38; 8.81; 8.22" in citation.snippet
    assert citation.snippet.endswith("2.5M; 2.3M")

    truncated = passage.to_citation(max_snippet_chars=95)
    assert len(truncated.snippet) <= 95
    assert truncated.snippet.endswith("…")
    assert "<br" not in truncated.snippet.lower()
