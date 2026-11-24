"""Tests for canonical `AttachedSource` contracts, Cache-Aside `LazyPdfIndexer`, and routing."""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import httpx
import pymupdf
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from qdrant_client import AsyncQdrantClient

from ai.agents.library.gateway.contracts import (
    ItemDetail,
    LibraryOverview,
    LibraryScope,
)
from ai.agents.library.module import build_library_module
from ai.agents.rag.ingestion.chunker import MarkdownStructureChunker
from ai.agents.rag.ingestion.lazy_indexer import (
    HttpFileContentFetcher,
    LazyPdfIndexer,
)
from ai.agents.rag.ingestion.parsers import build_default_parser_registry
from ai.agents.rag.ingestion.pipeline import IngestionPipeline
from ai.agents.rag.module import build_rag_module
from ai.orchestrator.contracts import AgentRegistry, AgentSpec
from ai.orchestrator.prompts import build_supervisor_prompt
from ai.orchestrator.state import RootState
from ai.shared.contracts.auth import bind_delegation_token
from ai.shared.contracts.knowledge import KnowledgeDocumentInput
from ai.shared.contracts.sources import (
    AttachedSource,
    collect_rag_source_ids,
    parse_attached_sources,
)
from ai.shared.llm.embeddings import DeterministicHashEmbedder
from ai.shared.llm.factory import ChatRunnable
from ai.shared.persistence.knowledge import QdrantKnowledgeStore


def _make_pdf_bytes(*, title: str, text: str) -> bytes:
    doc = pymupdf.open()
    doc.set_metadata({"title": title})
    page = doc.new_page()
    page.insert_text((72, 72), text)
    raw = doc.tobytes()
    doc.close()
    return raw


def _scripted_llm(responses: Sequence[str]) -> ChatRunnable:
    it = iter(responses)

    def _reply(_msg: LanguageModelInput) -> AIMessage:
        return AIMessage(content=next(it))

    return RunnableLambda(_reply)


def test_attached_source_separation_and_metadata_exclusion() -> None:
    raw = [
        {"kind": "upload", "title": "notes.md", "source_id": "upload-1"},
        {
            "kind": "library",
            "title": "Indexed Paper",
            "item_id": "item-indexed",
            "file_id": "file-indexed",
            "source_id": "file-indexed",
            "has_pdf": True,
            "indexed": True,
        },
        {
            "kind": "library",
            "title": "Unindexed PDF Paper",
            "item_id": "item-pdf",
            "file_id": "file-pdf",
            "has_pdf": True,
            "indexed": False,
        },
        {
            "kind": "library",
            "title": "Metadata Only Paper",
            "item_id": "item-meta",
            "has_pdf": False,
            "indexed": False,
        },
        # Duplicate entry should be deduped cleanly
        {"kind": "upload", "title": "notes.md", "source_id": "upload-1"},
        # Malformed entry should be ignored gracefully
        {"kind": "unknown-kind"},
    ]

    parsed = parse_attached_sources(raw)
    assert len(parsed) == 4
    assert parsed[0].effective_source_id == "upload-1"
    assert parsed[1].effective_source_id == "file-indexed"
    assert parsed[2].effective_source_id == "file-pdf"
    assert parsed[3].effective_source_id is None

    # Metadata-only item_id must NEVER leak into Qdrant filters even if in fallback documentIds
    doc_ids = collect_rag_source_ids(parsed, fallback_document_ids=["item-meta", "extra-upload"])
    assert doc_ids == ("upload-1", "file-indexed", "file-pdf", "extra-upload")


async def test_qdrant_has_source_is_space_isolated() -> None:
    qdrant = AsyncQdrantClient(location=":memory:")
    try:
        store = QdrantKnowledgeStore(qdrant, DeterministicHashEmbedder(dimensions=64))
        assert await store.has_source(space_id="sp-1", source_id="file-1") is False

        await store.upsert(
            KnowledgeDocumentInput(
                space_id="sp-1",
                source_type="file",
                source_id="file-1",
                title="Paper",
                content="Sparse attention improves memory efficiency by 60 percent.",
            )
        )
        assert await store.has_source(space_id="sp-1", source_id="file-1") is True
        assert await store.has_source(space_id="sp-2", source_id="file-1") is False
        assert await store.has_source(space_id="sp-1", source_id="file-missing") is False
    finally:
        await qdrant.close()


async def test_lazy_pdf_indexer_cache_aside_limits_and_error_isolation() -> None:
    qdrant = AsyncQdrantClient(location=":memory:")
    try:
        store = QdrantKnowledgeStore(qdrant, DeterministicHashEmbedder(dimensions=64))
        pipeline = IngestionPipeline(
            parser_registry=build_default_parser_registry(),
            chunker=MarkdownStructureChunker(),
            knowledge_store=store,
        )

        fetch_calls: list[str] = []

        class _FakeFetcher:
            async def fetch_bytes(self, file_id: str) -> tuple[bytes, str, str]:
                fetch_calls.append(file_id)
                if file_id == "file-broken":
                    raise RuntimeError("Storage object corrupt")
                return (
                    _make_pdf_bytes(
                        title=f"Title {file_id}",
                        text=f"Findings for {file_id}: KV cache reduced memory by 52%.",
                    ),
                    f"{file_id}.pdf",
                    "application/pdf",
                )

        indexer = LazyPdfIndexer(
            ingestion_pipeline=pipeline,
            knowledge_store=store,
            file_fetcher=_FakeFetcher(),
            max_per_turn=2,
        )

        sources = [
            AttachedSource(
                kind="library",
                title="Paper One",
                item_id="item-1",
                file_id="file-1",
                has_pdf=True,
            ),
            AttachedSource(
                kind="library",
                title="Metadata Only",
                item_id="item-meta",
                has_pdf=False,
            ),
            AttachedSource(
                kind="library",
                title="Broken PDF",
                item_id="item-broken",
                file_id="file-broken",
                has_pdf=True,
            ),
            AttachedSource(
                kind="library",
                title="Paper Two",
                item_id="item-2",
                file_id="file-2",
                has_pdf=True,
            ),
            AttachedSource(
                kind="library",
                title="Paper Three Over Cap",
                item_id="item-3",
                file_id="file-3",
                has_pdf=True,
            ),
        ]

        # Turn 1: misses Qdrant -> lazily indexes file-1 & file-2, skips metadata, isolates broken, caps file-3
        report1 = await indexer.ensure_indexed(space_id="sp-lazy", sources=sources)
        assert report1.ready_source_ids == ("file-1", "file-2")
        assert report1.newly_indexed == ("file-1", "file-2")
        assert report1.skipped_metadata_only == ("item-meta",)
        assert len(report1.failed) == 2
        assert report1.failed[0]["source_id"] == "file-broken"
        assert report1.failed[1]["error"] == "lazy_index_turn_limit_reached"
        assert fetch_calls == ["file-1", "file-broken", "file-2"]

        # Verify Qdrant payload stores canonical source_type="file" and item_id metadata
        hits = await store.search(
            space_id="sp-lazy",
            query="KV cache memory",
            source_ids=["file-1"],
        )
        assert len(hits) >= 1
        assert hits[0].source_type == "file"
        assert hits[0].metadata.get("item_id") == "item-1"
        assert hits[0].metadata.get("file_id") == "file-1"

        # Turn 2: Cache-Aside hit for file-1 & file-2 -> ZERO fetches for those two
        fetch_calls.clear()
        report2 = await indexer.ensure_indexed(
            space_id="sp-lazy",
            sources=[sources[0], sources[3]],
        )
        assert report2.ready_source_ids == ("file-1", "file-2")
        assert report2.newly_indexed == ()
        assert fetch_calls == []
    finally:
        await qdrant.close()


async def test_http_file_fetcher_strips_auth_on_presigned_redirect() -> None:
    seen_requests: list[tuple[str, bool]] = []
    pdf_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"

    def _handler(request: httpx.Request) -> httpx.Response:
        has_auth = "authorization" in {k.lower() for k in request.headers}
        seen_requests.append((str(request.url), has_auth))
        if request.url.host == "localhost":
            return httpx.Response(
                302,
                headers={
                    "Location": "https://r2.cloudflare.example/bucket/obj?X-Amz-Signature=sig123"
                },
            )
        assert not has_auth, "Authorization header must be stripped on cross-origin S3/R2 redirect"
        return httpx.Response(
            200,
            content=pdf_bytes,
            headers={
                "Content-Type": "application/pdf",
                "Content-Disposition": 'attachment; filename="attention_paper.pdf"',
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
        fetcher = HttpFileContentFetcher(
            backend_url="http://localhost:3000",
            http_client=client,
        )
        with bind_delegation_token("secret-user-jwt"):
            raw, filename, mime = await fetcher.fetch_bytes("blob-42")

    assert raw == pdf_bytes
    assert filename == "attention_paper.pdf"
    assert mime == "application/pdf"
    assert seen_requests == [
        ("http://localhost:3000/api/v1/files/blob-42/content", True),
        ("https://r2.cloudflare.example/bucket/obj?X-Amz-Signature=sig123", False),
    ]


async def test_rag_agent_lazy_indexes_library_pdf_and_abstains_metadata_only() -> None:
    qdrant = AsyncQdrantClient(location=":memory:")
    try:
        store = QdrantKnowledgeStore(qdrant, DeterministicHashEmbedder(dimensions=64))
        fetched: list[str] = []

        class _StubFetcher:
            async def fetch_bytes(self, file_id: str) -> tuple[bytes, str, str]:
                fetched.append(file_id)
                return (
                    _make_pdf_bytes(
                        title="Flux Attention Paper",
                        text="Flux Attention achieves 94.2% recall on LongBench with linear complexity.",
                    ),
                    "flux_attention.pdf",
                    "application/pdf",
                )

        llm = _scripted_llm(
            [
                json.dumps(
                    {
                        "standalone_query": "Flux Attention recall on LongBench",
                        "search_queries": ["Flux Attention recall LongBench"],
                    }
                ),
                "Flux Attention đạt 94.2% recall trên LongBench với độ phức tạp tuyến tính [1].",
            ]
        )

        rag_mod = build_rag_module(
            llm=llm,
            knowledge_store=store,
            allow_remote_rerank=False,
            file_fetcher=_StubFetcher(),
        )

        # Unindexed Library PDF: source_id=None at attach time, resolved & indexed lazily via file_id
        state: RootState = {
            "messages": [HumanMessage(content="Recall trên LongBench là bao nhiêu?")],
            "context": {
                "attached_sources": [
                    {
                        "kind": "library",
                        "title": "Flux Attention Paper",
                        "item_id": "item-99",
                        "file_id": "file-99",
                        "has_pdf": True,
                        "indexed": False,
                    }
                ]
            },
        }
        out = await rag_mod.agent_spec.handler(state, {"configurable": {"space_id": "sp-e2e"}})
        assert out.status == "ok"
        assert "94.2%" in out.content
        assert fetched == ["file-99"]
        assert out.artifacts.get("lazy_indexed_count") == 1
        assert len(out.citations) == 1
        assert out.citations[0].source_id == "file-99"

        # Metadata-only attachment must abstain cleanly instead of running unfiltered workspace search
        meta_state: RootState = {
            "messages": [HumanMessage(content="Nội dung chương 3 nói gì?")],
            "context": {
                "attached_sources": [
                    {
                        "kind": "library",
                        "title": "Abstract Only Paper",
                        "item_id": "item-meta",
                        "has_pdf": False,
                        "indexed": False,
                    }
                ]
            },
        }
        meta_out = await rag_mod.agent_spec.handler(
            meta_state,
            {"configurable": {"space_id": "sp-e2e"}},
        )
        assert meta_out.artifacts.get("abstained") is True
        assert "metadata" in meta_out.content.lower()
    finally:
        await qdrant.close()


async def test_library_agent_uses_attached_item_ids_directly() -> None:
    seen_prompts: list[str] = []

    class _FakeGateway:
        def __init__(self) -> None:
            self.got_ids: list[str] = []

        @property
        def read(self) -> _FakeGateway:
            return self

        @property
        def write(self) -> _FakeGateway:
            return self

        async def search_items(
            self,
            scope: LibraryScope,
            *,
            query: str = "",
            year_from: int | None = None,
            year_to: int | None = None,
            item_type: str | None = None,
            collection_id: str | None = None,
            tag: str | None = None,
            read_status: str | None = None,
            limit: int = 20,
        ) -> dict[str, Any]:
            raise AssertionError(
                f"search_items should not be called when item_id is attached: {query}"
            )

        async def get_overview(self, scope: LibraryScope) -> LibraryOverview:
            return LibraryOverview()

        async def get_item(
            self,
            scope: LibraryScope,
            item_id: str,
            *,
            include_notes: bool = True,
            include_relations: bool = True,
        ) -> ItemDetail:
            del scope, include_notes, include_relations
            self.got_ids.append(item_id)
            return ItemDetail(
                id=item_id,
                title="Attention Is All You Need",
                authors=("Ashish Vaswani",),
                year=2017,
                abstract="The dominant sequence transduction models are based on transformers.",
            )

        async def audit_health(
            self,
            scope: LibraryScope,
            *,
            check: Literal["all", "duplicates", "quality", "retractions"] = "all",
        ) -> dict[str, Any]:
            return {}

        async def cite_items(
            self,
            scope: LibraryScope,
            *,
            item_ids: Sequence[str] = (),
            style: str = "apa-7th",
            export_format: str | None = None,
        ) -> dict[str, Any]:
            return {}

        async def add_paper(
            self,
            scope: LibraryScope,
            *,
            source: Literal["doi", "arxiv", "pmid", "isbn", "url"],
            identifier: str = "",
            url: str | None = None,
            collection_id: str | None = None,
        ) -> dict[str, Any]:
            return {}

        async def manage_collection(
            self,
            scope: LibraryScope,
            *,
            action: Literal["create", "rename", "assign", "move", "detach"],
            collection_id: str | None = None,
            name: str | None = None,
            parent_id: str | None = None,
            item_ids: Sequence[str] = (),
            source_collection_id: str | None = None,
        ) -> dict[str, Any]:
            return {}

        async def organize_item(
            self,
            scope: LibraryScope,
            *,
            item_id: str,
            tags: Sequence[str] | None = None,
            read_status: str | None = None,
            is_starred: bool | None = None,
            link_item_ids: Sequence[str] = (),
            unlink_item_id: str | None = None,
        ) -> dict[str, Any]:
            return {}

        async def save_note(
            self,
            scope: LibraryScope,
            *,
            note_id: str | None = None,
            item_id: str | None = None,
            title: str | None = None,
            content_md: str = "",
            tags: Sequence[str] | None = None,
        ) -> dict[str, Any]:
            return {}

        async def trash_items(
            self, scope: LibraryScope, *, item_ids: Sequence[str]
        ) -> dict[str, Any]:
            return {}

        async def merge_duplicates(
            self,
            scope: LibraryScope,
            *,
            primary_item_id: str,
            duplicate_item_ids: Sequence[str],
        ) -> dict[str, Any]:
            return {}

    gateway = _FakeGateway()
    responses = iter(
        [
            json.dumps(
                {
                    "action": "call_tools",
                    "tool_calls": [
                        {
                            "tool": "get_item",
                            "args": {"item_id": "item-direct-7"},
                            "reason": "Read attached paper",
                        }
                    ],
                }
            ),
            json.dumps({"action": "respond", "answer": ""}),
            "Bài Attention Is All You Need (2017) đề xuất kiến trúc Transformer.",
        ]
    )

    def _recording_llm(messages: LanguageModelInput) -> AIMessage:
        if isinstance(messages, list):
            for m in messages:
                if isinstance(m, BaseMessage):
                    seen_prompts.append(str(m.content))
        return AIMessage(content=next(responses))

    lib_mod = build_library_module(
        llm=RunnableLambda(_recording_llm),
        gateway=gateway,
    )
    out = await lib_mod.agent_spec.handler(
        {
            "messages": [HumanMessage(content="Xem abstract của bài tôi vừa đính kèm")],
            "context": {
                "attached_sources": [
                    {
                        "kind": "library",
                        "title": "Attention Is All You Need",
                        "item_id": "item-direct-7",
                        "has_pdf": False,
                        "indexed": False,
                    }
                ]
            },
        },
        {"configurable": {"space_id": "sp-1"}},
    )
    assert out.status == "ok"
    assert "Transformer" in out.content
    assert gateway.got_ids == ["item-direct-7"]
    assert any("item-direct-7" in p and "Attached library items:" in p for p in seen_prompts)


def test_supervisor_prompt_renders_attached_sources_with_routing_guidance() -> None:
    async def _noop(state: RootState, cfg: Any) -> Any:
        raise NotImplementedError

    registry = AgentRegistry(
        [
            AgentSpec(name="rag_agent", description="Search indexed PDFs", handler=_noop),
            AgentSpec(name="library_agent", description="Manage Library items", handler=_noop),
        ]
    )
    prompt = build_supervisor_prompt(
        registry.format_catalog(),
        request_context={
            "document_ids": ["file-1", "item-meta", "extra-upload"],
            "attached_sources": [
                {
                    "kind": "library",
                    "title": "Transformer Survey",
                    "item_id": "item-pdf",
                    "file_id": "file-1",
                    "source_id": "file-1",
                    "has_pdf": True,
                    "indexed": False,
                },
                {
                    "kind": "library",
                    "title": "Metadata Note",
                    "item_id": "item-meta",
                    "has_pdf": False,
                    "indexed": False,
                },
            ],
        },
    )
    assert '[library] "Transformer Survey"' in prompt
    assert "status=PDF auto-indexable" in prompt
    assert "status=metadata/notes only" in prompt
    assert "extra-upload" in prompt
    # Already-covered file-1 & item-meta are not duplicated in fallback document_ids
    assert "Attached space document_ids: [extra-upload]" in prompt


def test_rag_slice_isolation_and_single_root_init() -> None:
    rag_root = Path("src/ai/agents/rag")
    for py_file in rag_root.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("ai.agents.library"), py_file
                assert not node.module.startswith("ai.agents.web_search"), py_file
    assert [p.as_posix() for p in Path("src/ai").rglob("__init__.py")] == ["src/ai/__init__.py"]
