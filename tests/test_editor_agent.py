"""Pure-DI unit and FastAPI integration tests for the 5-subsystem `editor_agent`."""

from __future__ import annotations

import ast
import json
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi.testclient import TestClient
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr

from ai.agents.editor.context.builder import (
    build_editor_context_snapshot,
    format_context_for_prompt,
    normalize_compile_errors,
)
from ai.agents.editor.context.citation_audit import audit_citation_gaps
from ai.agents.editor.context.preamble import extract_preamble_summary
from ai.agents.editor.context.structure_audit import audit_page_structure
from ai.agents.editor.gateway.contracts import (
    EditorCitationEntry,
    EditorDocument,
    EditorFileInfo,
    EditorGatewayError,
    EditorReviewComment,
    EditorReviewSuggestion,
    EditorScope,
    EditorSearchHit,
    EditorSymbolCatalog,
)
from ai.agents.editor.gateway.http import HttpEditorGateway
from ai.agents.editor.module import build_editor_module
from ai.agents.editor.patching.matcher import check_patch_overlap, resolve_patch
from ai.agents.editor.patching.validator import validate_patched_latex
from ai.main import create_app
from ai.orchestrator.contracts import AgentSpec
from ai.shared.config.database import DatabaseConfig, PostgresConfig, QdrantConfig, RedisConfig
from ai.shared.config.editor import EditorConfig
from ai.shared.config.embedding import EmbeddingConfig
from ai.shared.config.llm import LLMConfig
from ai.shared.config.server import AppEnvironment, ServerConfig
from ai.shared.config.settings import Settings
from ai.shared.contracts.auth import bind_delegation_token, get_delegation_token
from ai.shared.llm.factory import ChatRunnable


def _test_settings(*, hitl_policy: str = "all_writes") -> Settings:
    return Settings(
        server=ServerConfig(
            app_env=AppEnvironment.TEST,
            internal_api_key=SecretStr("test-key"),
        ),
        llm=LLMConfig(api_key=SecretStr("test-llm-key")),
        embedding=EmbeddingConfig(dimensions=64, collection_name="test_knowledge"),
        editor=EditorConfig(hitl_policy=hitl_policy),  # type: ignore[arg-type]
        database=DatabaseConfig(
            postgres=PostgresConfig(database_url=SecretStr("")),
            redis=RedisConfig(redis_url=SecretStr("")),
            qdrant=QdrantConfig(qdrant_url=""),
        ),
    )


def _scripted_llm(outputs: list[str]) -> ChatRunnable:
    idx = {"n": 0}

    def _respond(_messages: LanguageModelInput) -> AIMessage:
        pos = min(idx["n"], len(outputs) - 1)
        idx["n"] += 1
        return AIMessage(content=outputs[pos])

    return RunnableLambda(_respond)


class _ScriptedStreamingChatModel(BaseChatModel):
    step_chunks: list[list[str]]
    call_idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-editor-streaming-stub"

    def _next_chunks(self) -> list[str]:
        pos = min(self.call_idx, len(self.step_chunks) - 1)
        self.call_idx += 1
        return self.step_chunks[pos]

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        text = "".join(self._next_chunks())
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del messages, stop, run_manager, kwargs
        text = "".join(self._next_chunks())
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        del messages, stop, kwargs
        for tok in self._next_chunks():
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=tok))
            if run_manager:
                run_manager.on_llm_new_token(tok, chunk=chunk)
            yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        del messages, stop, kwargs
        for tok in self._next_chunks():
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=tok))
            if run_manager:
                await run_manager.on_llm_new_token(tok, chunk=chunk)
            yield chunk


class InMemoryFakeEditorGateway:
    """In-memory adapter implementing EditorGateway for deterministic tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tokens_seen: list[str] = []
        self.files: dict[str, dict[str, Any]] = {
            "doc-main": {
                "id": "doc-main",
                "path": "main.tex",
                "title": "main.tex",
                "is_root": True,
                "content": (
                    "\\documentclass{article}\n"
                    "\\usepackage{amsmath}\n"
                    "\\begin{document}\n"
                    "\\section{Introduction}\n"
                    "We study FluxKV compression.\n"
                    "\\input{sections/method}\n"
                    "\\end{document}"
                ),
            },
            "doc-method": {
                "id": "doc-method",
                "path": "sections/method.tex",
                "title": "method.tex",
                "is_root": False,
                "content": (
                    "\\section{Method}\\label{sec:method}\n"
                    "Our method reduces KV memory by half.\n"
                    "As shown in earlier work, eviction matters."
                ),
            },
            "doc-bib": {
                "id": "doc-bib",
                "path": "references.bib",
                "title": "references.bib",
                "is_root": False,
                "content": "@inproceedings{vaswani2017,\n  title={Attention Is All You Need},\n  year={2017}\n}",
            },
        }
        self.citations = [
            EditorCitationEntry(
                key="vaswani2017",
                entry_type="inproceedings",
                title="Attention Is All You Need",
                authors="Ashish Vaswani",
                year="2017",
                file_path="references.bib",
            )
        ]
        self.review_comments: list[EditorReviewComment] = [
            EditorReviewComment(
                id="cmt-1",
                status="open",
                author="Reviewer A",
                content="Please clarify the eviction threshold in Section 2.",
                line=3,
                line_end=3,
            )
        ]
        self.review_suggestions: list[EditorReviewSuggestion] = [
            EditorReviewSuggestion(
                id="sug-existing-1",
                status="pending",
                change_type="insert",
                author="Co-author",
                original_text="",
                suggested_text=" (see Appendix A)",
                from_line=2,
                to_line=2,
            )
        ]
        self.added_bibtex: list[dict[str, Any]] = []
        self.resolved_identifiers: list[str] = []
        self.synced_collections: list[str] = []
        self.created_suggestions: list[dict[str, Any]] = []
        self.created_pages: list[dict[str, Any]] = []

    def _record(self, op: str, **kwargs: Any) -> None:
        self.calls.append((op, kwargs))
        self.tokens_seen.append(get_delegation_token())

    async def list_files(
        self,
        scope: EditorScope,
        *,
        include_content: bool = False,
    ) -> list[EditorFileInfo]:
        self._record("list_files", page_id=scope.page_id, include_content=include_content)
        return [
            EditorFileInfo(
                id=f["id"],
                path=f["path"],
                title=f["title"],
                is_root=f["is_root"],
                content=f["content"] if include_content else None,
            )
            for f in self.files.values()
        ]

    async def read_file(self, scope: EditorScope, doc_id: str) -> EditorDocument:
        self._record("read_file", page_id=scope.page_id, doc_id=doc_id)
        item = self.files.get(doc_id)
        if item is None:
            for candidate in self.files.values():
                if candidate["path"] == doc_id:
                    item = candidate
                    break
        if item is None:
            raise EditorGatewayError(404, f"Document {doc_id} not found")
        return EditorDocument(
            id=item["id"],
            path=item["path"],
            title=item["title"],
            content=item["content"],
            rev=1,
            line_count=len(item["content"].splitlines()),
        )

    async def get_symbols(self, scope: EditorScope) -> EditorSymbolCatalog:
        self._record("get_symbols", page_id=scope.page_id)
        return EditorSymbolCatalog(
            root_doc_path="main.tex",
            labels=("sec:method",),
            citations=("vaswani2017",),
            packages=("amsmath",),
            files=[{"path": f["path"]} for f in self.files.values()],
        )

    async def search_pages(
        self,
        scope: EditorScope,
        *,
        query: str,
        case_sensitive: bool = False,
        whole_word: bool = False,
        use_regex: bool = False,
        file_ids: Sequence[str] = (),
        max_results: int = 30,
    ) -> list[EditorSearchHit]:
        del case_sensitive, whole_word, use_regex, file_ids, max_results
        self._record("search_pages", page_id=scope.page_id, query=query)
        hits: list[EditorSearchHit] = []
        for f in self.files.values():
            for line_idx, text in enumerate(f["content"].splitlines(), start=1):
                if query.lower() in text.lower():
                    hits.append(
                        EditorSearchHit(
                            file_id=f["id"],
                            file_path=f["path"],
                            line=line_idx,
                            preview=text,
                        )
                    )
        return hits

    async def list_citations(
        self,
        scope: EditorScope,
        *,
        query: str = "",
        limit: int = 100,
    ) -> list[EditorCitationEntry]:
        self._record("list_citations", page_id=scope.page_id, query=query)
        q = query.strip().lower()
        return [c for c in self.citations if not q or q in c.key.lower() or q in c.title.lower()][
            :limit
        ]

    async def validate_citations(self, scope: EditorScope) -> dict[str, Any]:
        self._record("validate_citations", page_id=scope.page_id)
        return {"valid": True, "duplicateKeys": [], "missingMetadata": []}

    async def lint_pages(
        self,
        scope: EditorScope,
        *,
        content: str | None = None,
        file_path: str | None = None,
    ) -> dict[str, Any]:
        self._record("lint_pages", page_id=scope.page_id, file_path=file_path)
        return {"errors": [], "warnings": [], "checkedChars": len(content or "")}

    async def list_review_comments(
        self,
        scope: EditorScope,
        *,
        doc_id: str | None = None,
        status: str | None = None,
    ) -> list[EditorReviewComment]:
        self._record("list_review_comments", page_id=scope.page_id, doc_id=doc_id, status=status)
        if status in {"open", "resolved"}:
            return [c for c in self.review_comments if c.status == status]
        return list(self.review_comments)

    async def list_review_suggestions(
        self,
        scope: EditorScope,
        *,
        doc_id: str | None = None,
        status: str | None = None,
    ) -> list[EditorReviewSuggestion]:
        self._record("list_review_suggestions", page_id=scope.page_id, doc_id=doc_id, status=status)
        if status in {"pending", "accepted", "rejected"}:
            return [s for s in self.review_suggestions if s.status == status]
        return list(self.review_suggestions)

    async def add_bibtex_entries(
        self,
        scope: EditorScope,
        *,
        raw_bibtex: str | None = None,
        entries: Sequence[dict[str, Any]] = (),
        target_bib_file: str = "references.bib",
    ) -> dict[str, Any]:
        self._record(
            "add_bibtex_entries",
            page_id=scope.page_id,
            target_bib_file=target_bib_file,
        )
        record = {
            "raw_bibtex": raw_bibtex,
            "entries": list(entries),
            "target_bib_file": target_bib_file,
        }
        self.added_bibtex.append(record)
        added_entries: list[dict[str, Any]] = []
        for entry in entries:
            key = str(entry.get("key") or entry.get("citeKey") or "added2024").strip()
            self.citations.append(
                EditorCitationEntry(
                    key=key,
                    entry_type=str(entry.get("entryType") or "article"),
                    title=str(entry.get("title") or "Untitled"),
                    year=str(entry.get("year") or "2024"),
                    file_path=target_bib_file,
                )
            )
            added_entries.append({"key": key})
        if raw_bibtex and not added_entries:
            added_entries.append({"key": "bibtex_added_2024"})
        return {"targetBibFile": target_bib_file, "entries": added_entries}

    async def resolve_citation(
        self,
        scope: EditorScope,
        *,
        identifier: str,
        target_bib_file: str = "references.bib",
    ) -> dict[str, Any]:
        self._record(
            "resolve_citation",
            page_id=scope.page_id,
            identifier=identifier,
            target_bib_file=target_bib_file,
        )
        self.resolved_identifiers.append(identifier)
        key = "arxiv1706_03762" if "1706" in identifier else "resolved_2024"
        self.citations.append(
            EditorCitationEntry(
                key=key,
                entry_type="article",
                title="Resolved Reference",
                year="2024",
                file_path=target_bib_file,
            )
        )
        return {
            "resolved": True,
            "entry": {"key": key, "title": "Resolved Reference"},
            "targetBibFile": target_bib_file,
        }

    async def sync_library_citations(
        self,
        scope: EditorScope,
        *,
        collection_id: str,
        target_bib_file: str = "references.bib",
    ) -> dict[str, Any]:
        self._record(
            "sync_library_citations",
            page_id=scope.page_id,
            collection_id=collection_id,
            target_bib_file=target_bib_file,
        )
        self.synced_collections.append(collection_id)
        return {
            "synced": 2,
            "entries": [{"key": "lib_sync_a"}, {"key": "lib_sync_b"}],
            "targetBibFile": target_bib_file,
        }

    async def create_review_suggestion(
        self,
        scope: EditorScope,
        *,
        doc_id: str,
        change_type: Literal["insert", "delete"],
        suggested_text: str = "",
        original_text: str = "",
        from_line: int,
        from_column: int,
        to_line: int,
        to_column: int,
    ) -> dict[str, Any]:
        self._record(
            "create_review_suggestion",
            page_id=scope.page_id,
            doc_id=doc_id,
            change_type=change_type,
        )
        payload = {
            "id": f"sug-{len(self.created_suggestions) + 1}",
            "docId": doc_id,
            "type": change_type,
            "suggestedText": suggested_text,
            "originalText": original_text,
            "fromLine": from_line,
            "fromColumn": from_column,
            "toLine": to_line,
            "toColumn": to_column,
            "status": "pending",
        }
        self.created_suggestions.append(payload)
        return {"suggestion": payload}

    async def create_page(
        self,
        scope: EditorScope,
        *,
        title: str,
        description: str | None = None,
        content: str | None = None,
        template_type: str | None = None,
    ) -> dict[str, Any]:
        self._record(
            "create_page",
            space_id=scope.space_id,
            title=title,
            template_type=template_type,
        )
        if not scope.space_id.strip():
            raise EditorGatewayError(400, "Active `space_id` is required to create a new page")
        payload = {
            "id": f"page-{len(self.created_pages) + 1}",
            "spaceId": scope.space_id,
            "title": title,
            "description": description,
            "content": content,
            "templateType": template_type,
            "status": "draft",
        }
        self.created_pages.append(payload)
        return payload


def _parse_sse(raw_body: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for frame in raw_body.strip().split("\n\n"):
        if not frame.strip():
            continue
        event_name = ""
        data_raw = ""
        for line in frame.splitlines():
            if line.startswith("event: "):
                event_name = line[7:].strip()
            elif line.startswith("data: "):
                data_raw = line[6:].strip()
        if event_name and data_raw:
            events.append((event_name, json.loads(data_raw)))
    return events


def test_context_preamble_viewport_and_error_normalization() -> None:
    many_lines = [
        "\\documentclass[11pt]{article}",
        "\\usepackage{amsmath, booktabs}",
        "\\newcommand{\\norm}[1]{\\|#1\\|}",
        "\\begin{document}",
    ]
    for i in range(1, 301):
        many_lines.append(
            f"Line {i} content in paper." if i != 180 else "$x_i = y_i$ bad delimiter"
        )
    many_lines.append("\\end{document}")

    summary = extract_preamble_summary("\n".join(many_lines))
    assert summary.document_class == "article"
    assert summary.packages == ("amsmath", "booktabs")
    assert "\\norm" in summary.custom_macros
    assert summary.begin_document_line == 4

    snap = build_editor_context_snapshot(
        {
            "page_id": "page-1",
            "filename": "sections/eval.tex",
            "file_content": "\n".join(many_lines),
            "cursor_line": 184,
            "cursor_column": 12,
            "selection": "$x_i = y_i$",
            "selection_start_line": 184,
            "selection_end_line": 184,
            "compile_errors": [
                {"line": 184, "severity": "error", "message": "Missing $ inserted"},
            ],
        },
        viewport_radius_lines=10,
    )
    assert snap.is_truncated is True
    assert snap.viewport_start_line == 174
    assert snap.viewport_end_line == 194
    assert "L184: $x_i = y_i$ bad delimiter" in snap.viewport_numbered

    rendered = format_context_for_prompt(snap)
    assert "\\norm" in rendered
    assert "[ERROR] (sections/eval.tex:L184)" in rendered

    errs = normalize_compile_errors(
        {"errors": [{"line": 12, "message": "Undefined control sequence"}]}
    )
    assert len(errs) == 1
    assert errs[0].line == 12


def test_patching_matcher_and_validator_edge_cases() -> None:
    doc = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "First line\n"
        "Alpha   beta\n"
        "Repeated line\n"
        "Middle\n"
        "Repeated line\n"
        "\\end{document}\n"
    )

    # Exact unique replace
    p1, patched1, err1 = resolve_patch(
        doc,
        file_id="f1",
        file_path="main.tex",
        kind="replace",
        search_text="First line\nAlpha   beta",
        replace_text="First line\n\\alpha + \\beta",
        title="Convert to math",
    )
    assert err1 is None and p1 is not None
    assert p1.match_strategy == "exact"
    assert p1.from_line == 3 and p1.to_line == 4
    assert validate_patched_latex(doc, patched1).ok is True

    # Ambiguous match rejected with line hint
    p_amb, _, err_amb = resolve_patch(
        doc,
        file_id="f1",
        file_path="main.tex",
        kind="replace",
        search_text="Repeated line",
        replace_text="Unique line",
        title="Ambiguous",
    )
    assert p_amb is None and err_amb is not None and "[5, 7]" in err_amb

    # Whitespace-tolerant unique match returns verbatim canonical search_text
    p_ws, _, err_ws = resolve_patch(
        doc,
        file_id="f1",
        file_path="main.tex",
        kind="replace",
        search_text="  First line  \n  Alpha   beta  ",
        replace_text="Replaced",
        title="Trimmed match",
    )
    assert err_ws is None and p_ws is not None
    assert p_ws.match_strategy == "normalized_whitespace"
    assert p_ws.search_text == "First line\nAlpha   beta"

    # Line prefix hint
    p_pref, _, err_pref = resolve_patch(
        doc,
        file_id="f1",
        file_path="main.tex",
        kind="replace",
        search_text="L003: First line",
        replace_text="Changed",
        title="Line prefix",
    )
    assert p_pref is None and err_pref is not None and "L###:" in err_pref

    # insert_preamble
    p_pre, patched_pre, err_pre = resolve_patch(
        doc,
        file_id="f1",
        file_path="main.tex",
        kind="insert_preamble",
        search_text="",
        replace_text="\\usepackage{booktabs}",
        title="Add booktabs",
    )
    assert err_pre is None and p_pre is not None
    assert p_pre.kind == "insert_preamble"
    assert "\\usepackage{booktabs}\n\\begin{document}" in patched_pre

    # Duplicate preamble rejected
    p_dup, _, err_dup = resolve_patch(
        patched_pre,
        file_id="f1",
        file_path="main.tex",
        kind="insert_preamble",
        search_text="",
        replace_text="\\usepackage{booktabs}",
        title="Duplicate",
    )
    assert p_dup is None and err_dup is not None and "already present" in err_dup.lower()

    # Syntax imbalance detection (tolerant of pre-existing errors elsewhere)
    broken = doc.replace("Alpha   beta", "\\begin{align}\nx = 1")
    res_bad = validate_patched_latex(doc, broken)
    assert res_bad.ok is False
    assert any("Unbalanced LaTeX environment" in e for e in res_bad.errors)

    # Escaped braces in verbatim/backslash-parity & existing error tolerance
    doc_with_preexisting = doc + "\n\\begin{itemize}\n"
    good_local = validate_patched_latex(
        doc_with_preexisting,
        doc_with_preexisting.replace("First line", "Updated \\% {ok}"),
    )
    assert good_local.ok is True

    # Citation and label warnings when authoritative sets provided
    cited_doc = doc.replace("Alpha   beta", "See \\cite{hallucinated2025} and \\ref{fig:missing}.")
    sym_res = validate_patched_latex(
        doc,
        cited_doc,
        known_cite_keys=frozenset({"vaswani2017"}),
        known_labels=frozenset({"sec:method"}),
    )
    assert sym_res.ok is True
    assert len(sym_res.warnings) == 2

    # Overlap guard
    overlap = check_patch_overlap([p1], p_ws)
    assert overlap is not None and "Overlapping patch rejected" in overlap


async def test_http_editor_gateway_unwraps_envelope_and_forwards_auth() -> None:
    seen_headers: list[tuple[str, str, dict[str, Any] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode()) if request.content else None
        seen_headers.append((request.method, request.url.path, body))
        auth = request.headers.get("authorization", "")
        if auth != "Bearer editor-jwt":
            return httpx.Response(401, json={"message": "Unauthorized"})

        if request.url.path.endswith("/files"):
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "files": [
                            {
                                "id": "d1",
                                "path": "/main.tex",
                                "title": "main.tex",
                                "isRootDoc": True,
                            }
                        ]
                    },
                },
            )
        if request.url.path.endswith("/lint-doc"):
            return httpx.Response(
                200, json={"success": True, "data": {"errors": [], "warnings": []}}
            )
        if request.url.path.endswith("/search"):
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "results": [
                            {
                                "fileId": "d1",
                                "fileName": "main.tex",
                                "matches": [
                                    {"line": 4, "text": "\\section{Intro}", "matchStart": 0}
                                ],
                            }
                        ]
                    },
                },
            )
        return httpx.Response(404, json={"message": "Not found"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gw = HttpEditorGateway(base_url="http://backend.local", http_client=client)
        scope = EditorScope(page_id="page-99", active_file_path="main.tex")
        with bind_delegation_token("Bearer editor-jwt"):
            files = await gw.list_files(scope)
            lint_res = await gw.lint_pages(
                scope, content="\\documentclass{article}", file_path="main.tex"
            )
            hits = await gw.search_pages(scope, query="Intro")

    assert files[0].path == "main.tex"
    assert files[0].is_root is True
    assert lint_res == {"errors": [], "warnings": []}
    assert seen_headers[1][2] == {
        "content": "\\documentclass{article}",
        "filename": "main.tex",
        "pageId": "page-99",
    }
    assert hits[0].line == 4 and hits[0].file_path == "main.tex"


def test_editor_agent_self_corrects_and_streams_patch_artifact() -> None:
    fake_gw = InMemoryFakeEditorGateway()
    router_json = json.dumps(
        {
            "mode": "delegate",
            "targets": [
                {"agent": "editor_agent", "instruction": "Cite Vaswani in sections/method.tex"}
            ],
            "needs_followup": False,
        }
    )
    step1_read = json.dumps(
        {
            "action": "call_tools",
            "tool_calls": [
                {"tool": "read_file", "args": {"file_path": "sections/method.tex"}},
                {"tool": "inspect_citations", "args": {"query": "vaswani"}},
            ],
        }
    )
    step2_bad_anchor = json.dumps(
        {
            "action": "call_tools",
            "tool_calls": [
                {
                    "tool": "propose_edit",
                    "args": {
                        "file_path": "sections/method.tex",
                        "kind": "replace",
                        "search_text": "Non existent line in file",
                        "replace_text": "Should fail first",
                        "title": "Bad attempt",
                    },
                }
            ],
        }
    )
    step3_good_anchor = json.dumps(
        {
            "action": "call_tools",
            "tool_calls": [
                {
                    "tool": "propose_edit",
                    "args": {
                        "file_path": "sections/method.tex",
                        "kind": "replace",
                        "search_text": "As shown in earlier work, eviction matters.",
                        "replace_text": "As shown in earlier work~\\cite{vaswani2017}, eviction matters.",
                        "title": "Add Vaswani citation",
                        "rationale": "Grounded in references.bib",
                    },
                }
            ],
        }
    )
    router_stub = _ScriptedStreamingChatModel(step_chunks=[[router_json]])
    agent_stub = _ScriptedStreamingChatModel(
        step_chunks=[
            [step1_read],
            [step2_bad_anchor],
            [step3_good_anchor],
            ["Proposed ", "citation patch ", "ready."],
        ]
    )

    app = create_app(
        settings=_test_settings(),
        llm=agent_stub,
        router_llm=router_stub,
        editor_gateway=fake_gw,
    )
    with TestClient(app) as client:
        res = client.post(
            "/chat",
            headers={"X-Internal-Key": "test-key", "Authorization": "Bearer jwt-ed-1"},
            json={
                "messages": [{"role": "user", "content": "Cite Vaswani in sections/method.tex"}],
                "space_id": "space-1",
                "chat_id": "ed-chat-1",
                "page_id": "page-123",
                "context": {
                    "active_file_id": "doc-main",
                    "active_file_path": "main.tex",
                    "file_content": fake_gw.files["doc-main"]["content"],
                },
            },
        )
    assert res.status_code == 200
    parsed = _parse_sse(res.text)
    deltas = [data["delta"] for name, data in parsed if name == "text.delta"]
    assert "".join(deltas) == "Proposed citation patch ready."

    finished_name, finished_payload = parsed[-1]
    assert finished_name == "run.finished"
    assert finished_payload["route"] == "editor_agent"
    patches = finished_payload["artifacts"]["editor_patches"]
    assert len(patches) == 1
    assert patches[0]["file_path"] == "sections/method.tex"
    assert patches[0]["file_id"] == "doc-method"
    assert patches[0]["from_line"] == 3
    assert "\\cite{vaswani2017}" in patches[0]["replace_text"]
    assert patches[0]["warnings"] == []
    assert "jwt-ed-1" in fake_gw.tokens_seen


def test_editor_agent_sync_and_peer_handoff() -> None:
    fake_gw = InMemoryFakeEditorGateway()

    async def fake_lib_handler(_state: Any, _config: Any) -> dict[str, Any]:
        return {
            "content": "Found Vaswani 2017 (key `vaswani2017`) in user library.",
            "status": "ok",
        }

    router_stub = _scripted_llm(
        [
            json.dumps(
                {
                    "mode": "delegate",
                    "targets": [{"agent": "library_agent", "instruction": "Find Attention paper"}],
                    "needs_followup": True,
                }
            ),
            json.dumps(
                {
                    "mode": "delegate",
                    "targets": [
                        {"agent": "editor_agent", "instruction": "Draft citation sentence"}
                    ],
                    "needs_followup": False,
                }
            ),
        ]
    )
    editor_module = build_editor_module(
        llm=_scripted_llm(
            [
                json.dumps(
                    {
                        "action": "call_tools",
                        "tool_calls": [
                            {
                                "tool": "propose_edit",
                                "args": {
                                    "kind": "insert_preamble",
                                    "replace_text": "\\usepackage{booktabs}",
                                    "title": "Add booktabs package",
                                },
                            }
                        ],
                    }
                ),
                "Added booktabs to preamble.",
            ]
        ),
        gateway=fake_gw,
    )
    from ai.orchestrator.contracts import AgentRegistry

    registry = AgentRegistry(
        [
            AgentSpec(name="library_agent", description="Fake lib", handler=fake_lib_handler),
            editor_module.agent_spec,
        ]
    )
    synth_llm = _scripted_llm(["Added booktabs using peer library findings."])

    app = create_app(
        settings=_test_settings(),
        llm=synth_llm,
        router_llm=router_stub,
        registry=registry,
        editor_gateway=fake_gw,
    )
    with TestClient(app) as client:
        res = client.post(
            "/chat/sync",
            headers={"X-Internal-Key": "test-key", "Authorization": "Bearer sync-jwt"},
            json={
                "messages": [{"role": "user", "content": "Find Attention paper and update LaTeX"}],
                "space_id": "s1",
                "chat_id": "ed-sync-1",
                "page_id": "page-sync",
                "context": {
                    "filename": "main.tex",
                    "file_content": fake_gw.files["doc-main"]["content"],
                },
            },
        )
    assert res.status_code == 200
    body = res.json()
    assert body["route"] == "library_agent->editor_agent"
    assert len(body["artifacts"]["editor_patches"]) == 1
    assert body["artifacts"]["editor_patches"][0]["kind"] == "insert_preamble"


def test_default_bootstrap_registers_editor_agent_and_slice_isolation() -> None:
    app = create_app(settings=_test_settings(), llm=_scripted_llm(["ok"]))
    with TestClient(app) as client:
        agents_res = client.get("/agents", headers={"X-Internal-Key": "test-key"})
    assert agents_res.status_code == 200
    names = [item["name"] for item in agents_res.json()["agents"]]
    assert "editor_agent" in names

    ed_root = Path("src/ai/agents/editor")
    assert (ed_root / "module.py").is_file()
    assert list(ed_root.rglob("__init__.py")) == []

    forbidden = ("ai.agents.rag", "ai.agents.web_search", "ai.agents.library")
    for py_file in ed_root.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(forbidden), f"{py_file} imports {node.module}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(forbidden), f"{py_file} imports {alias.name}"


def test_citation_audit_detects_undefined_keys_unused_entries_and_unsupported_claims() -> None:
    tex = (
        "\\begin{abstract}\n"
        "Prior work showed improvements in decoding speed.\n"
        "\\end{abstract}\n"
        "\\section{Introduction}\n"
        "We build upon standard transformers\\cite{vaswani2017, missing_key}.\n"
        "Prior work has shown that KV cache quantization hurts tail accuracy.\n"
        "Our method reduces KV memory by half without degradation.\n"
        "\\begin{equation}\n"
        "y = W x\n"
        "\\end{equation}\n"
        "\\section{Evaluation}\n"
        "Sparse eviction achieves a 35\\% speedup over dense decoding.\n"
    )
    report = audit_citation_gaps(
        tex,
        file_path="main.tex",
        known_bib_keys=frozenset({"vaswani2017", "unused_ref_2022"}),
    )
    assert report.file_path == "main.tex"
    assert report.total_citations_used == 2
    assert report.total_bib_entries == 2
    assert report.undefined_keys == ("missing_key",)
    assert report.unused_bib_keys == ("unused_ref_2022",)
    flagged = [c.sentence for c in report.unsupported_claims]
    assert any("KV cache quantization hurts tail accuracy" in s for s in flagged)
    assert any("35\\% speedup" in s for s in flagged)
    assert not any("decoding speed" in s for s in flagged)
    assert not any("standard transformers" in s for s in flagged)
    assert not any("Our method reduces" in s for s in flagged)


async def test_http_editor_gateway_phase2_review_and_citation_routes() -> None:
    seen: list[tuple[str, str, dict[str, Any] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] | None = None
        if request.content:
            body = json.loads(request.content.decode("utf-8"))
        seen.append((request.method, str(request.url), body))
        path = request.url.path
        if path == "/api/v1/pages/pages/doc-method/comments" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "comments": [
                            {
                                "id": "c-1",
                                "status": "open",
                                "author": {"name": "Reviewer B"},
                                "content": "Add baseline citation.",
                                "line": 4,
                                "lineEnd": 4,
                            }
                        ]
                    },
                },
            )
        if path == "/api/v1/pages/pages/doc-method/suggestions" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {
                        "suggestions": [
                            {
                                "id": "s-1",
                                "status": "pending",
                                "type": "insert",
                                "author": {"name": "Alice"},
                                "originalText": "",
                                "suggestedText": " \\cite{vaswani2017}",
                                "fromLine": 4,
                                "toLine": 4,
                            }
                        ]
                    },
                },
            )
        if path == "/api/v1/pages/pages/page-1/citations/entries" and request.method == "POST":
            return httpx.Response(
                200,
                json={"success": True, "data": {"entries": [{"key": "dao2022"}]}},
            )
        if path == "/api/v1/pages/pages/page-1/citations/resolve" and request.method == "POST":
            return httpx.Response(
                200,
                json={"success": True, "data": {"entry": {"key": "arxiv1706"}}},
            )
        if path == "/api/v1/pages/pages/page-1/citations/sync-library" and request.method == "POST":
            return httpx.Response(
                200,
                json={"success": True, "data": {"synced": 3, "entries": [{"key": "lib1"}]}},
            )
        if path == "/api/v1/pages/pages/doc-method/suggestions" and request.method == "POST":
            return httpx.Response(
                201,
                json={"success": True, "data": {"suggestion": {"id": "s-new"}}},
            )
        return httpx.Response(404, json={"message": f"Unexpected route {path}"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        gw = HttpEditorGateway(
            base_url="http://backend.local",
            http_client=client,
            token_provider=lambda: "tok-phase2",
        )
        scope = EditorScope(
            page_id="page-1",
            space_id="space-9",
            active_file_id="doc-method",
            active_file_path="sections/method.tex",
        )

        comments = await gw.list_review_comments(scope, status="open")
        assert len(comments) == 1
        assert comments[0].author == "Reviewer B"
        assert comments[0].line == 4

        suggestions = await gw.list_review_suggestions(scope, status="pending")
        assert len(suggestions) == 1
        assert suggestions[0].change_type == "insert"

        add_res = await gw.add_bibtex_entries(
            scope,
            raw_bibtex="@inproceedings{dao2022, title={FlashAttention}}",
            target_bib_file="refs.bib",
        )
        assert add_res["entries"][0]["key"] == "dao2022"

        res_res = await gw.resolve_citation(scope, identifier="1706.03762")
        assert res_res["entry"]["key"] == "arxiv1706"

        sync_res = await gw.sync_library_citations(scope, collection_id="col-42")
        assert sync_res["synced"] == 3

        sug_res = await gw.create_review_suggestion(
            scope,
            doc_id="doc-method",
            change_type="insert",
            suggested_text=" \\cite{vaswani2017}",
            original_text="",
            from_line=3,
            from_column=41,
            to_line=3,
            to_column=41,
        )
        assert sug_res["suggestion"]["id"] == "s-new"

    post_bodies = {
        url.split("/api/v1/pages")[1]: body
        for m, url, body in seen
        if m == "POST" and body is not None
    }
    assert post_bodies["/pages/page-1/citations/entries"]["targetBibFile"] == "refs.bib"
    assert post_bodies["/pages/page-1/citations/resolve"]["identifier"] == "1706.03762"
    assert post_bodies["/pages/page-1/citations/sync-library"]["collectionId"] == "col-42"
    sug_body = post_bodies["/pages/doc-method/suggestions"]
    assert sug_body["fromLine"] == 3 and sug_body["fromColumn"] == 41
    assert sug_body["spaceId"] == "space-9"


def test_editor_agent_hitl_resolve_citation_approve_and_reject_via_fastapi() -> None:
    from ai.orchestrator.contracts import AgentRegistry

    headers = {"X-Internal-Key": "test-key", "Authorization": "Bearer jwt-ed-hitl"}

    approve_gw = InMemoryFakeEditorGateway()
    approve_llm = _scripted_llm(
        [
            json.dumps(
                {
                    "action": "call_tools",
                    "tool_calls": [
                        {
                            "tool": "resolve_citation",
                            "args": {
                                "identifier": "1706.03762",
                                "target_bib_file": "references.bib",
                            },
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "action": "respond",
                    "answer": "Added arXiv:1706.03762 (`arxiv1706_03762`) to references.bib.",
                }
            ),
        ]
    )
    approve_mod = build_editor_module(llm=approve_llm, gateway=approve_gw)
    approve_app = create_app(
        settings=_test_settings(),
        llm=approve_llm,
        registry=AgentRegistry([approve_mod.agent_spec]),
        editor_gateway=approve_gw,
    )
    with TestClient(approve_app) as client:
        first = client.post(
            "/chat/sync",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "Resolve arXiv:1706.03762 into bib"}],
                "space_id": "s1",
                "chat_id": "ed-hitl-approve",
                "page_id": "page-1",
                "intent_hint": "editor_agent",
                "context": {
                    "page_id": "page-1",
                    "filename": "main.tex",
                    "file_content": approve_gw.files["doc-main"]["content"],
                },
            },
        )
        assert first.status_code == 200
        first_body = first.json()
        assert first_body["status"] == "interrupted"
        intr = first_body["interrupts"][0]
        assert intr["tool"] == "resolve_citation"
        assert "1706.03762" in intr["description"]
        assert approve_gw.resolved_identifiers == []

        resumed = client.post(
            "/chat/resume/sync",
            headers=headers,
            json={
                "decision": [{"decision": "approve", "interruptId": intr["id"]}],
                "space_id": "s1",
                "chat_id": "ed-hitl-approve",
            },
        )
        assert resumed.status_code == 200
        resumed_body = resumed.json()
        assert resumed_body["status"] == "completed"
        assert "arxiv1706_03762" in resumed_body["content"]
        assert approve_gw.resolved_identifiers == ["1706.03762"]

    reject_gw = InMemoryFakeEditorGateway()
    reject_llm = _scripted_llm(
        [
            json.dumps(
                {
                    "action": "call_tools",
                    "tool_calls": [
                        {
                            "tool": "resolve_citation",
                            "args": {
                                "identifier": "10.1145/999999",
                                "target_bib_file": "references.bib",
                            },
                        }
                    ],
                }
            ),
            "Citation import was cancelled by user confirmation.",
        ]
    )
    reject_mod = build_editor_module(llm=reject_llm, gateway=reject_gw)
    reject_app = create_app(
        settings=_test_settings(),
        llm=reject_llm,
        registry=AgentRegistry([reject_mod.agent_spec]),
        editor_gateway=reject_gw,
    )
    with TestClient(reject_app) as client:
        p1 = client.post(
            "/chat/sync",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "Resolve DOI 10.1145/999999"}],
                "space_id": "s1",
                "chat_id": "ed-hitl-reject",
                "page_id": "page-1",
                "intent_hint": "editor_agent",
                "context": {
                    "page_id": "page-1",
                    "filename": "main.tex",
                    "file_content": reject_gw.files["doc-main"]["content"],
                },
            },
        )
        assert p1.status_code == 200
        intr2 = p1.json()["interrupts"][0]
        p2 = client.post(
            "/chat/resume/sync",
            headers=headers,
            json={
                "decision": [{"decision": "reject", "interruptId": intr2["id"]}],
                "space_id": "s1",
                "chat_id": "ed-hitl-reject",
            },
        )
        assert p2.status_code == 200
        assert p2.json()["status"] == "completed"
        assert "cancelled" in p2.json()["content"].lower()
        assert reject_gw.resolved_identifiers == []


def test_editor_agent_scan_citation_gaps_and_create_review_suggestion_coordinates() -> None:
    from ai.orchestrator.contracts import AgentRegistry

    gw = InMemoryFakeEditorGateway()
    llm = _scripted_llm(
        [
            json.dumps(
                {
                    "action": "call_tools",
                    "tool_calls": [
                        {
                            "tool": "scan_citation_gaps",
                            "args": {"file_path": "sections/method.tex"},
                        },
                        {
                            "tool": "list_review_comments",
                            "args": {"file_path": "sections/method.tex", "status": "open"},
                        },
                    ],
                }
            ),
            json.dumps(
                {
                    "action": "call_tools",
                    "tool_calls": [
                        {
                            "tool": "create_review_suggestion",
                            "args": {
                                "file_path": "sections/method.tex",
                                "change_type": "insert",
                                "search_text": "As shown in earlier work, eviction matters.",
                                "suggested_text": " \\cite{vaswani2017}",
                                "rationale": "Back prior-work claim flagged by Citation Reviewer",
                            },
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "action": "respond",
                    "answer": "Created a Track Changes suggestion adding \\cite{vaswani2017}.",
                }
            ),
        ]
    )
    mod = build_editor_module(llm=llm, gateway=gw)
    app = create_app(
        settings=_test_settings(),
        llm=llm,
        registry=AgentRegistry([mod.agent_spec]),
        editor_gateway=gw,
    )
    headers = {"X-Internal-Key": "test-key", "Authorization": "Bearer jwt-ed-sug"}

    with TestClient(app) as client:
        res1 = client.post(
            "/chat/sync",
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "Review citations and suggest fixes"}],
                "space_id": "s1",
                "chat_id": "ed-sug-flow",
                "page_id": "page-1",
                "intent_hint": "editor_agent",
                "context": {
                    "page_id": "page-1",
                    "active_file_id": "doc-method",
                    "filename": "sections/method.tex",
                    "file_content": gw.files["doc-method"]["content"],
                },
            },
        )
        assert res1.status_code == 200
        b1 = res1.json()
        assert b1["status"] == "interrupted"
        intr = b1["interrupts"][0]
        assert intr["tool"] == "create_review_suggestion"
        op_names = [op for op, _ in gw.calls]
        assert "list_review_comments" in op_names
        assert "list_review_suggestions" in op_names

        res2 = client.post(
            "/chat/resume/sync",
            headers=headers,
            json={
                "decision": [{"decision": "approve", "interruptId": intr["id"]}],
                "space_id": "s1",
                "chat_id": "ed-sug-flow",
            },
        )
        assert res2.status_code == 200
        b2 = res2.json()
        assert b2["status"] == "completed"
        assert len(gw.created_suggestions) == 1
        sug = gw.created_suggestions[0]
        assert sug["docId"] == "doc-method"
        assert sug["type"] == "insert"
        # "As shown in earlier work, eviction matters." is L3, col 43 at end of line
        assert sug["fromLine"] == 3 and sug["toLine"] == 3
        assert sug["fromColumn"] == 43 and sug["toColumn"] == 43
        assert sug["suggestedText"] == " \\cite{vaswani2017}"


def test_structure_audit_multi_file_outline_refs_includes_and_packages() -> None:
    files = {
        "main.tex": (
            "\\documentclass{article}\n"
            "\\usepackage{amsmath}\n"
            "% \\usepackage{booktabs} commented out intentionally\n"
            "\\begin{document}\n"
            "\\section{Introduction}\\label{sec:intro}\n"
            "We describe the pipeline cleanly.\n"
            "\\input{sections/method}\n"
            "\\input{sections/missing_eval}\n"
            "See also \\ref{sec:method} and \\cref{eq:missing, sec:orphan}.\n"
            "\\begin{verbatim}\n"
            "\\includegraphics{fake.png}\n"
            "\\label{sec:intro}\n"
            "\\end{verbatim}\n"
            "\\end{document}\n"
        ),
        "sections/method.tex": (
            "\\section{Method}\\label{sec:method}\n"
            "Our objective aligns across blocks with fast runtime.\n"
            "\\subsection{Overview}\\label{sec:method}\n"
            "\\includegraphics{arch.png}\n"
            "\\begin{tabular}{ll}\\toprule A & B \\\\\\bottomrule\\end{tabular}\n"
        ),
        "sections/orphan.tex": (
            "\\section{Orphan Notes}\\label{sec:orphan}\n"
            "Standalone draft file never included by main.\n"
        ),
    }
    report = audit_page_structure(
        files,
        root_file_hint="main.tex",
        include_unreferenced_labels=True,
    )
    assert report.root_file == "main.tex"
    assert report.total_tex_files == 3
    assert report.total_words_approx > 10
    assert [(n.file_path, n.level, n.title) for n in report.outline] == [
        ("main.tex", "section", "Introduction"),
        ("sections/method.tex", "section", "Method"),
        ("sections/method.tex", "subsection", "Overview"),
        ("sections/orphan.tex", "section", "Orphan Notes"),
    ]
    assert report.outline[0].approx_words >= 5
    # Duplicate label only counted outside verbatim block (sections/method.tex L1 & L3)
    assert len(report.duplicate_labels) == 1
    assert report.duplicate_labels[0].label == "sec:method"
    assert report.duplicate_labels[0].locations == (
        "sections/method.tex:L1",
        "sections/method.tex:L3",
    )
    # Undefined ref detects eq:missing while resolved sec:method / sec:orphan are valid
    assert [(r.command, r.key) for r in report.undefined_refs] == [("\\cref", "eq:missing")]
    assert report.unreferenced_labels == ("sec:intro",)
    assert [(m.file_path, m.target) for m in report.missing_includes] == [
        ("main.tex", "sections/missing_eval")
    ]
    assert report.orphan_tex_files == ("sections/orphan.tex",)
    # Package checks ignore verbatim \includegraphics and commented \usepackage{booktabs}
    assert report.loaded_packages == ("amsmath",)
    missing_pkgs = {m.package: (m.file_path, m.line) for m in report.missing_packages}
    assert missing_pkgs == {
        "cleveref": ("main.tex", 9),
        "graphicx": ("sections/method.tex", 4),
        "booktabs": ("sections/method.tex", 5),
    }

    # Standalone snippet without \documentclass should not emit orphan or package false positives
    snippet_report = audit_page_structure(
        {
            "sections/a.tex": "\\section{A}\\label{sec:a}\nHello world.\n",
            "sections/b.tex": "\\section{B}\\label{sec:b}\n\\includegraphics{fig.png}\n",
        }
    )
    assert snippet_report.orphan_tex_files == ()
    assert snippet_report.missing_packages == ()


def test_editor_agent_streams_live_tool_and_patch_preview_events_before_text() -> None:
    fake_gw = InMemoryFakeEditorGateway()
    router_json = json.dumps(
        {
            "mode": "delegate",
            "targets": [
                {
                    "agent": "editor_agent",
                    "instruction": "Audit page structure and cite Vaswani in sections/method.tex",
                }
            ],
            "needs_followup": False,
        }
    )
    step1_audit = json.dumps(
        {
            "action": "call_tools",
            "tool_calls": [{"tool": "audit_page_structure", "args": {}}],
        }
    )
    step2_patch = json.dumps(
        {
            "action": "call_tools",
            "tool_calls": [
                {
                    "tool": "propose_edit",
                    "args": {
                        "file_path": "sections/method.tex",
                        "kind": "replace",
                        "search_text": "As shown in earlier work, eviction matters.",
                        "replace_text": "As shown in earlier work~\\cite{vaswani2017}, eviction matters.",
                        "title": "Add Vaswani citation",
                    },
                }
            ],
        }
    )
    router_stub = _ScriptedStreamingChatModel(step_chunks=[[router_json]])
    agent_stub = _ScriptedStreamingChatModel(
        step_chunks=[
            [step1_audit],
            [step2_patch],
            ["Structure ", "audited ", "cleanly."],
        ]
    )

    app = create_app(
        settings=_test_settings(),
        llm=agent_stub,
        router_llm=router_stub,
        editor_gateway=fake_gw,
    )
    with TestClient(app) as client:
        res = client.post(
            "/chat",
            headers={"X-Internal-Key": "test-key", "Authorization": "Bearer jwt-ed-p3"},
            json={
                "messages": [{"role": "user", "content": "Audit page and propose cite fix"}],
                "space_id": "space-p3",
                "chat_id": "ed-p3-stream",
                "page_id": "page-123",
                "context": {
                    "active_file_id": "doc-main",
                    "active_file_path": "main.tex",
                    "file_content": fake_gw.files["doc-main"]["content"],
                },
            },
        )
    assert res.status_code == 200
    parsed = _parse_sse(res.text)
    event_names = [name for name, _ in parsed]

    called_tools = [data["tool"] for name, data in parsed if name == "tool.called"]
    completed_tools = [data["tool"] for name, data in parsed if name == "tool.completed"]
    assert "audit_page_structure" in called_tools
    assert "propose_edit" in called_tools
    assert "audit_page_structure" in completed_tools
    assert "propose_edit" in completed_tools

    previews = [data for name, data in parsed if name == "editor.patch_preview"]
    assert len(previews) == 1
    assert previews[0]["agent"] == "editor_agent"
    assert previews[0]["total_patches"] == 1
    assert previews[0]["patch"]["file_path"] == "sections/method.tex"
    assert "\\cite{vaswani2017}" in previews[0]["patch"]["replace_text"]

    preview_idx = event_names.index("editor.patch_preview")
    first_text_idx = event_names.index("text.delta")
    finished_idx = event_names.index("run.finished")
    assert preview_idx < first_text_idx < finished_idx

    deltas = [data["delta"] for name, data in parsed if name == "text.delta"]
    assert "".join(deltas) == "Structure audited cleanly."
    _, finished_payload = parsed[-1]
    assert len(finished_payload["artifacts"]["editor_patches"]) == 1


async def test_editor_design_patterns_extensibility() -> None:
    from ai.agents.editor.patching.contracts import MatchStrategy
    from ai.agents.editor.patching.matcher import DEFAULT_ANCHOR_STRATEGIES, find_unique_anchor
    from ai.agents.editor.patching.validator import DEFAULT_VALIDATION_RULES, ValidationContext
    from ai.agents.editor.tools.registry import build_default_editor_tool_registry
    from ai.agents.editor.workflow.contracts import EditorAgentConfig
    from ai.agents.editor.workflow.events import EditorEventPublisher, NullEditorEvents
    from ai.agents.editor.workflow.nodes import build_editor_nodes

    class _ForbidTodoRule:
        def check(self, ctx: ValidationContext) -> tuple[tuple[str, ...], tuple[str, ...]]:
            if "TODO:" in ctx.patched_clean and "TODO:" not in ctx.orig_clean:
                return ("Unresolved TODO marker introduced.",), ()
            return (), ()

    val_res = validate_patched_latex(
        "\\section{Intro}\nClean text.\n",
        "\\section{Intro}\nTODO: finish text.\n",
        rules=(*DEFAULT_VALIDATION_RULES, _ForbidTodoRule()),
    )
    assert val_res.ok is False
    assert val_res.errors == ("Unresolved TODO marker introduced.",)

    class _CaseFoldAnchorStrategy:
        name: MatchStrategy = "normalized_whitespace"

        def locate(self, content: str, search_text: str) -> list[tuple[int, int]]:
            idx = content.lower().find(search_text.lower())
            return [(idx, idx + len(search_text))] if idx >= 0 else []

    span, strategy_used, err = find_unique_anchor(
        "Title: FluxKV\nBody text.",
        "TITLE: FLUXKV",
        strategies=(*DEFAULT_ANCHOR_STRATEGIES, _CaseFoldAnchorStrategy()),
    )
    assert err is None and span == (0, 13) and strategy_used == "normalized_whitespace"

    registry = build_default_editor_tool_registry()
    resolve_spec = registry.get("resolve_citation")
    suggest_spec = registry.get("create_review_suggestion")
    assert resolve_spec is not None and suggest_spec is not None
    assert (
        resolve_spec.format_confirmation({"identifier": "10.1/x", "target_bib_file": "refs.bib"})
        == "Resolve academic identifier `10.1/x` and append BibTeX to `refs.bib`"
    )
    assert (
        suggest_spec.format_confirmation({"change_type": "delete", "file_path": "main.tex"})
        == "Create Track Changes `delete` suggestion in `main.tex`"
    )

    pub: EditorEventPublisher = NullEditorEvents()
    gw = InMemoryFakeEditorGateway()
    nodes = build_editor_nodes(
        llm=_scripted_llm(["ok"]),
        gateway=gw,
        tool_registry=registry,
        agent_config=EditorAgentConfig(),
        event_publisher=pub,
    )
    assert set(nodes) == {
        "hydrate_context",
        "plan_and_act",
        "policy_gate",
        "execute_and_verify",
        "synthesize_response",
    }
    res_exec = await nodes["execute_and_verify"](
        {
            "page_id": "page-1",
            "approved_calls": [{"tool": "list_files", "args": {"include_content": False}}],
            "observations": [],
        },
        {},
    )
    assert res_exec["observations"][0]["status"] == "ok"


def test_preamble_order_validation_rejects_usepackage_before_documentclass() -> None:
    orig = "\\documentclass{article}\n\\begin{document}\n\\author{Tấn Thành}\n\\end{document}\n"

    bad_separate = validate_patched_latex(
        orig,
        "\\usepackage[T1]{fontenc}\n\\documentclass{article}\n\\begin{document}\n\\author{Tấn Thành}\n\\end{document}\n",
    )
    assert bad_separate.ok is False
    assert any("before `\\documentclass`" in err for err in bad_separate.errors)

    bad_concat = validate_patched_latex(
        orig,
        "\\usepackage[T1]{fontenc}\\documentclass{article}\n\\begin{document}\n\\author{Tấn Thành}\n\\end{document}\n",
    )
    assert bad_concat.ok is False
    assert any("before `\\documentclass`" in err for err in bad_concat.errors)

    good_direct = validate_patched_latex(
        orig,
        "\\documentclass{article}\n\\usepackage[T5]{fontenc}\n\\begin{document}\n\\author{Tấn Thành}\n\\end{document}\n",
    )
    assert good_direct.ok is True

    proposal, patched, match_err = resolve_patch(
        orig,
        file_id="page-1",
        file_path="main.tex",
        kind="insert_preamble",
        search_text="",
        replace_text="\\usepackage[T5]{fontenc}",
        title="Add T5 fontenc",
    )
    assert match_err is None and proposal is not None
    good_resolved = validate_patched_latex(orig, patched)
    assert good_resolved.ok is True


async def test_http_editor_gateway_create_page_and_tool_without_open_page() -> None:
    import pytest

    from ai.agents.editor.tools.contracts import EditorToolRuntime
    from ai.agents.editor.tools.registry import build_default_editor_tool_registry
    from ai.agents.editor.tools.write import CreatePageArgs, handle_create_page

    captured: list[tuple[str, str, dict[str, Any] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode()) if request.content else None
        captured.append((request.method, request.url.path, body))
        if request.url.path == "/api/v1/pages/spaces/space-1/pages" and request.method == "POST":
            return httpx.Response(
                201,
                json={
                    "success": True,
                    "data": {
                        "id": "page-new-1",
                        "spaceId": "space-1",
                        "title": (body or {}).get("title"),
                        "status": "draft",
                    },
                },
            )
        return httpx.Response(404, json={"message": "Not found"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gw = HttpEditorGateway(base_url="http://backend.local", http_client=client)
        with bind_delegation_token("Bearer editor-jwt"):
            with pytest.raises(EditorGatewayError, match="space_id"):
                await gw.create_page(EditorScope(page_id="", space_id=""), title="FFA")

            created = await gw.create_page(
                EditorScope(page_id="", space_id="space-1"),
                title="FFA",
                description="Flux for Academic article",
                template_type="article",
            )

    assert created["id"] == "page-new-1"
    assert created["title"] == "FFA"
    assert captured == [
        (
            "POST",
            "/api/v1/pages/spaces/space-1/pages",
            {
                "title": "FFA",
                "description": "Flux for Academic article",
                "templateType": "article",
            },
        )
    ]

    fake_gw = InMemoryFakeEditorGateway()
    registry = build_default_editor_tool_registry()
    spec = registry.get("create_page")
    assert spec is not None and spec.risk_tier == "write"
    assert "create_page" in spec.format_confirmation(
        {"title": "FFA"}
    ) or "FFA" in spec.format_confirmation({"title": "FFA"})

    runtime = EditorToolRuntime(
        gateway=fake_gw,
        scope=EditorScope(page_id="", space_id="space-1"),
    )
    res = await handle_create_page(
        runtime,
        CreatePageArgs(title="FFA", description="Flux for Academic"),
    )
    assert res.status == "ok"
    assert res.tool == "create_page"
    assert res.data is not None and res.data["title"] == "FFA"
    assert len(fake_gw.created_pages) == 1
