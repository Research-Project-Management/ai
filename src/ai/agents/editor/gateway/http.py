"""HTTP adapter implementing EditorGateway against NestJS Flux Pages APIs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal
from urllib.parse import quote

import httpx

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
from ai.agents.editor.gateway.mapping import (
    project_citation_entries,
    project_document,
    project_file_list,
    project_review_comments,
    project_review_suggestions,
    project_search_hits,
    project_symbol_catalog,
)
from ai.shared.contracts.auth import get_delegation_token
from ai.shared.gateway.transport import BackendTransport, encode_seg


def editor_page_route(page_id: str) -> str:
    """Return the REST path for a single page resource."""
    return f"/api/v1/pages/pages/{encode_seg(page_id)}"


class HttpEditorGateway:
    """Async REST adapter calling Flux Backend `/api/v1/pages/...` APIs with JWT delegation."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 15.0,
        http_client: httpx.AsyncClient | None = None,
        token_provider: Callable[[], str] = get_delegation_token,
        internal_api_key: str = "",
    ) -> None:
        self._transport = BackendTransport(
            base_url=base_url,
            service="Pages",
            error_factory=EditorGatewayError,
            timeout_seconds=timeout_seconds,
            http_client=http_client,
            token_provider=token_provider,
            internal_api_key=internal_api_key,
        )

    def _require_page_id(self, scope: EditorScope) -> str:
        clean = scope.page_id.strip()
        if not clean:
            raise EditorGatewayError(
                400,
                "Active page `page_id` is required to access page files",
            )
        return clean

    def _require_space_id(self, scope: EditorScope) -> str:
        clean = scope.space_id.strip()
        if not clean:
            raise EditorGatewayError(
                400,
                "Active `space_id` is required to create a new page",
            )
        return clean

    def _encoded_page_id(self, scope: EditorScope) -> str:
        return quote(self._require_page_id(scope), safe="")

    def _encoded_doc_id(self, scope: EditorScope, doc_id: str | None = None) -> str:
        target = (doc_id or scope.active_file_id or self._require_page_id(scope)).strip()
        return quote(target, safe="")

    async def list_files(
        self,
        scope: EditorScope,
        *,
        include_content: bool = False,
    ) -> list[EditorFileInfo]:
        page_id = self._encoded_page_id(scope)
        raw = await self._transport.request(
            "GET",
            f"/api/v1/pages/docs/{page_id}/files",
            params={"includeContent": "true" if include_content else "false"},
        )
        return project_file_list(raw, include_content=include_content)

    async def read_file(self, scope: EditorScope, doc_id: str) -> EditorDocument:
        del scope
        if not doc_id.strip():
            raise EditorGatewayError(400, "read_file requires a non-empty document ID")
        raw = await self._transport.request("GET", editor_page_route(doc_id))
        if not isinstance(raw, dict):
            raise EditorGatewayError(502, "Unexpected document payload format from Backend")
        return project_document(raw, doc_id=doc_id)

    async def get_symbols(self, scope: EditorScope) -> EditorSymbolCatalog:
        page_id = self._encoded_page_id(scope)
        raw = await self._transport.request(
            "GET", f"/api/v1/pages/pages/{page_id}/structure/symbols"
        )
        return project_symbol_catalog(raw)

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
        clean_query = query.strip()
        if not clean_query:
            return []
        page_id = self._encoded_page_id(scope)
        capped = max(1, min(max_results, 100))
        body: dict[str, Any] = {
            "query": clean_query,
            "caseSensitive": case_sensitive,
            "wholeWord": whole_word,
            "useRegex": use_regex,
            "maxResults": capped,
        }
        clean_ids = [str(fid).strip() for fid in file_ids if str(fid).strip()]
        if clean_ids:
            body["fileIds"] = clean_ids

        raw = await self._transport.request(
            "POST", f"/api/v1/pages/pages/{page_id}/search", json_body=body
        )
        return project_search_hits(raw, max_results=capped)

    async def list_citations(
        self,
        scope: EditorScope,
        *,
        query: str = "",
        limit: int = 100,
    ) -> list[EditorCitationEntry]:
        page_id = self._encoded_page_id(scope)
        capped = max(1, min(limit, 500))
        clean_query = query.strip()
        if clean_query:
            raw = await self._transport.request(
                "GET",
                f"/api/v1/pages/pages/{page_id}/citations/search",
                params={"query": clean_query, "limit": capped},
            )
        else:
            raw = await self._transport.request(
                "GET",
                f"/api/v1/pages/pages/{page_id}/citations",
                params={"limit": capped},
            )
        return project_citation_entries(raw, limit=capped)

    async def validate_citations(self, scope: EditorScope) -> dict[str, Any]:
        page_id = self._encoded_page_id(scope)
        raw = await self._transport.request(
            "GET", f"/api/v1/pages/pages/{page_id}/citations/validate"
        )
        return raw if isinstance(raw, dict) else {"result": raw}

    async def lint_pages(
        self,
        scope: EditorScope,
        *,
        content: str | None = None,
        file_path: str | None = None,
    ) -> dict[str, Any]:
        page_id = self._encoded_page_id(scope)
        if content is not None:
            filename = (file_path or scope.active_file_path or "main.tex").lstrip("/") or "main.tex"
            raw = await self._transport.request(
                "POST",
                f"/api/v1/pages/diagnostics/pages/{page_id}/lint-doc",
                json_body={"content": content, "filename": filename, "pageId": scope.page_id},
            )
        else:
            raw = await self._transport.request(
                "POST", f"/api/v1/pages/diagnostics/pages/{page_id}/lint"
            )
        return raw if isinstance(raw, dict) else {"result": raw}

    async def list_review_comments(
        self,
        scope: EditorScope,
        *,
        doc_id: str | None = None,
        status: str | None = None,
    ) -> list[EditorReviewComment]:
        encoded_id = self._encoded_doc_id(scope, doc_id)
        params: dict[str, Any] = {}
        if status and status.strip().lower() in {"open", "resolved"}:
            params["status"] = status.strip().lower()

        raw = await self._transport.request(
            "GET",
            f"/api/v1/pages/pages/{encoded_id}/comments",
            params=params or None,
        )
        return project_review_comments(raw)

    async def list_review_suggestions(
        self,
        scope: EditorScope,
        *,
        doc_id: str | None = None,
        status: str | None = None,
    ) -> list[EditorReviewSuggestion]:
        encoded_id = self._encoded_doc_id(scope, doc_id)
        params: dict[str, Any] = {}
        if status and status.strip().lower() in {"pending", "accepted", "rejected"}:
            params["status"] = status.strip().lower()

        raw = await self._transport.request(
            "GET",
            f"/api/v1/pages/pages/{encoded_id}/suggestions",
            params=params or None,
        )
        return project_review_suggestions(raw)

    async def add_bibtex_entries(
        self,
        scope: EditorScope,
        *,
        raw_bibtex: str | None = None,
        entries: Sequence[dict[str, Any]] = (),
        target_bib_file: str = "references.bib",
    ) -> dict[str, Any]:
        page_id = self._encoded_page_id(scope)
        body: dict[str, Any] = {
            "targetBibFile": target_bib_file.strip() or "references.bib",
            "pageId": scope.page_id,
        }
        if raw_bibtex and raw_bibtex.strip():
            body["rawBibtex"] = raw_bibtex.strip()
        if entries:
            body["entries"] = list(entries)
        raw = await self._transport.request(
            "POST",
            f"/api/v1/pages/pages/{page_id}/citations/entries",
            json_body=body,
        )
        return raw if isinstance(raw, dict) else {"result": raw}

    async def resolve_citation(
        self,
        scope: EditorScope,
        *,
        identifier: str,
        target_bib_file: str = "references.bib",
    ) -> dict[str, Any]:
        page_id = self._encoded_page_id(scope)
        body = {
            "identifier": identifier.strip(),
            "targetBibFile": target_bib_file.strip() or "references.bib",
            "pageId": scope.page_id,
        }
        raw = await self._transport.request(
            "POST",
            f"/api/v1/pages/pages/{page_id}/citations/resolve",
            json_body=body,
        )
        return raw if isinstance(raw, dict) else {"result": raw}

    async def sync_library_citations(
        self,
        scope: EditorScope,
        *,
        collection_id: str,
        target_bib_file: str = "references.bib",
    ) -> dict[str, Any]:
        page_id = self._encoded_page_id(scope)
        body = {
            "collectionId": collection_id.strip(),
            "targetBibFile": target_bib_file.strip() or "references.bib",
            "pageId": scope.page_id,
        }
        raw = await self._transport.request(
            "POST",
            f"/api/v1/pages/pages/{page_id}/citations/sync-library",
            json_body=body,
        )
        return raw if isinstance(raw, dict) else {"result": raw}

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
        encoded_id = quote(doc_id.strip(), safe="")
        body: dict[str, Any] = {
            "type": change_type,
            "suggestedText": suggested_text,
            "originalText": original_text,
            "fromLine": from_line,
            "fromColumn": from_column,
            "toLine": to_line,
            "toColumn": to_column,
        }
        if scope.space_id.strip():
            body["spaceId"] = scope.space_id.strip()
        raw = await self._transport.request(
            "POST",
            f"/api/v1/pages/pages/{encoded_id}/suggestions",
            json_body=body,
        )
        return raw if isinstance(raw, dict) else {"result": raw}

    async def create_page(
        self,
        scope: EditorScope,
        *,
        title: str,
        description: str | None = None,
        content: str | None = None,
        template_type: str | None = None,
    ) -> dict[str, Any]:
        space_id = quote(self._require_space_id(scope), safe="")
        body: dict[str, Any] = {"title": title.strip() or "Untitled Page"}
        if description is not None and description.strip():
            body["description"] = description.strip()
        if content is not None and content.strip():
            body["content"] = content
        if template_type is not None and template_type.strip():
            body["templateType"] = template_type.strip()
        raw = await self._transport.request(
            "POST",
            f"/api/v1/pages/spaces/{space_id}/pages",
            json_body=body,
        )
        return raw if isinstance(raw, dict) else {"result": raw}
