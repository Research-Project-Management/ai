"""Structure-aware Markdown chunker with table preservation and contextual headers."""

import re
from dataclasses import dataclass

import tiktoken

from ai.agents.rag.ingestion.contracts import ParsedDocument
from ai.shared.contracts.knowledge import KnowledgeChunkInput

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
_FENCE_START_RE = re.compile(r"^(```|~~~|\$\$)")


@dataclass(frozen=True)
class _Block:
    kind: str
    text: str
    section: str
    page_number: int | None


def _split_blocks(
    markdown: str,
    *,
    page_number: int | None,
    headings: dict[int, str],
) -> list[_Block]:
    blocks: list[_Block] = []
    lines = markdown.splitlines()
    idx = 0
    total = len(lines)

    def current_section() -> str:
        return " > ".join(headings[k] for k in sorted(headings))

    while idx < total:
        line = lines[idx]
        stripped = line.strip()
        if not stripped:
            idx += 1
            continue

        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            for existing_lvl in [k for k in headings if k >= level]:
                del headings[existing_lvl]
            headings[level] = title
            blocks.append(
                _Block(
                    kind="heading",
                    text=stripped,
                    section=current_section(),
                    page_number=page_number,
                )
            )
            idx += 1
            continue

        fence_match = _FENCE_START_RE.match(stripped)
        if fence_match:
            marker = fence_match.group(1)
            fence_lines = [line]
            idx += 1
            while idx < total:
                fence_lines.append(lines[idx])
                if lines[idx].strip().startswith(marker):
                    idx += 1
                    break
                idx += 1
            blocks.append(
                _Block(
                    kind="fence",
                    text="\n".join(fence_lines).strip(),
                    section=current_section(),
                    page_number=page_number,
                )
            )
            continue

        if stripped.startswith("|"):
            table_lines = [line]
            idx += 1
            while idx < total and lines[idx].strip().startswith("|"):
                table_lines.append(lines[idx])
                idx += 1
            blocks.append(
                _Block(
                    kind="table",
                    text="\n".join(table_lines).strip(),
                    section=current_section(),
                    page_number=page_number,
                )
            )
            continue

        prose_lines = [line]
        idx += 1
        while idx < total:
            nxt = lines[idx].strip()
            if not nxt or _HEADING_RE.match(nxt) or _FENCE_START_RE.match(nxt) or nxt.startswith("|"):
                break
            prose_lines.append(lines[idx])
            idx += 1
        blocks.append(
            _Block(
                kind="prose",
                text="\n".join(prose_lines).strip(),
                section=current_section(),
                page_number=page_number,
            )
        )

    return blocks


def _format_contextual_chunk(
    *,
    doc_title: str,
    section: str,
    body: str,
) -> str:
    header_parts: list[str] = []
    if doc_title:
        header_parts.append(f"Document: {doc_title}")
    if section:
        header_parts.append(f"Section: {section}")
    if not header_parts:
        return body
    return f"[{' | '.join(header_parts)}]\n{body}"


class MarkdownStructureChunker:
    """Structure-aware recursive chunker keeping headings, tables, and fenced blocks intact."""

    def __init__(
        self,
        *,
        chunk_size_tokens: int = 384,
        chunk_overlap_tokens: int = 48,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        self._chunk_size = chunk_size_tokens
        self._overlap = chunk_overlap_tokens
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count exact tokenizer tokens without tripping on special marker strings."""
        return len(self._encoding.encode(text, disallowed_special=()))

    def _split_long_prose(self, text: str) -> list[str]:
        tokens = self._encoding.encode(text, disallowed_special=())
        if len(tokens) <= self._chunk_size:
            return [text]
        pieces: list[str] = []
        step = max(1, self._chunk_size - self._overlap)
        for start in range(0, len(tokens), step):
            window = tokens[start : start + self._chunk_size]
            piece = self._encoding.decode(window).strip()
            if piece:
                pieces.append(piece)
            if start + self._chunk_size >= len(tokens):
                break
        return pieces

    def _split_oversize_table(self, table_text: str) -> list[str]:
        rows = [row for row in table_text.splitlines() if row.strip()]
        if len(rows) <= 3 or self.count_tokens(table_text) <= self._chunk_size * 2:
            return [table_text]

        header_rows = rows[:2] if "---" in rows[1] else rows[:1]
        data_rows = rows[len(header_rows) :]
        header_block = "\n".join(header_rows)
        header_tokens = self.count_tokens(header_block)

        splits: list[str] = []
        current_rows: list[str] = []
        current_tokens = header_tokens
        for row in data_rows:
            row_tokens = self.count_tokens(row) + 1
            if current_rows and current_tokens + row_tokens > self._chunk_size:
                splits.append("\n".join([*header_rows, *current_rows]))
                current_rows = [row]
                current_tokens = header_tokens + row_tokens
            else:
                current_rows.append(row)
                current_tokens += row_tokens

        if current_rows:
            splits.append("\n".join([*header_rows, *current_rows]))
        return splits or [table_text]

    def chunk(self, document: ParsedDocument, *, title: str = "") -> list[KnowledgeChunkInput]:
        doc_title = (title or document.title).strip()
        headings: dict[int, str] = {}
        all_blocks: list[_Block] = []
        for page in document.pages:
            all_blocks.extend(
                _split_blocks(
                    page.markdown,
                    page_number=page.page_number,
                    headings=headings,
                )
            )

        if not all_blocks:
            return []

        raw_chunks: list[tuple[int | None, str, str]] = []
        buf_parts: list[str] = []
        buf_page: int | None = None
        buf_section = ""

        def flush() -> None:
            nonlocal buf_parts, buf_page, buf_section
            body = "\n\n".join(part for part in buf_parts if part.strip()).strip()
            if body:
                raw_chunks.append((buf_page, buf_section, body))
            buf_parts = []
            buf_page = None
            buf_section = ""

        for block in all_blocks:
            if not block.text.strip():
                continue

            if buf_parts and (buf_page != block.page_number or buf_section != block.section):
                flush()

            if block.kind == "table":
                candidate = "\n\n".join([*buf_parts, block.text]) if buf_parts else block.text
                if buf_parts and self.count_tokens(candidate) > self._chunk_size:
                    flush()
                table_parts = self._split_oversize_table(block.text)
                if len(table_parts) > 1 or self.count_tokens(block.text) > self._chunk_size:
                    flush()
                    for part in table_parts:
                        raw_chunks.append((block.page_number, block.section, part))
                    continue

            if block.kind in {"prose", "fence"} and self.count_tokens(block.text) > self._chunk_size:
                flush()
                if block.kind == "fence":
                    raw_chunks.append((block.page_number, block.section, block.text))
                else:
                    for part in self._split_long_prose(block.text):
                        raw_chunks.append((block.page_number, block.section, part))
                continue

            candidate = "\n\n".join([*buf_parts, block.text]) if buf_parts else block.text
            if buf_parts and self.count_tokens(candidate) > self._chunk_size:
                overlap_tail = buf_parts[-1] if buf_parts[-1].startswith("#") else ""
                flush()
                if overlap_tail and block.kind != "heading":
                    buf_parts = [overlap_tail]
                    buf_page = block.page_number
                    buf_section = block.section

            if not buf_parts:
                buf_page = block.page_number
                buf_section = block.section
            buf_parts.append(block.text)

        flush()

        results: list[KnowledgeChunkInput] = []
        for idx, (page_num, section, body) in enumerate(raw_chunks):
            enriched = _format_contextual_chunk(
                doc_title=doc_title,
                section=section,
                body=body,
            )
            chunk_meta = {"section": section, "raw_content": body} if section else {"raw_content": body}
            results.append(
                KnowledgeChunkInput(
                    chunk_index=idx,
                    content=enriched,
                    page=page_num,
                    metadata=chunk_meta,
                )
            )
        return results
