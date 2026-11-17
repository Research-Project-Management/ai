"""System prompts for the abstract Supervisor planner and Synthesizer."""

from collections.abc import Mapping, Sequence
from typing import Any

from ai.orchestrator.citations import CITATION_INSTRUCTIONS, format_evidence_tag
from ai.orchestrator.contracts import AgentOutput
from ai.shared.context.contracts import ContextEnvelope
from ai.shared.contracts.sources import AttachedSource, parse_attached_sources


def _format_attached_source_line(src: AttachedSource) -> str:
    label = src.title or src.item_id or src.effective_source_id or "Untitled"
    if src.kind in ("library", "paper"):
        if src.indexed:
            status = "full-text indexed"
        elif src.has_pdf or src.file_id:
            status = "PDF auto-indexable"
        else:
            status = "metadata/notes only"
        item_ref = src.item_id or "none"
        sid_ref = src.effective_source_id or "none"
        return (
            f'  - [library] "{label}" '
            f"(item_id={item_ref}, status={status}, source_id={sid_ref})"
        )
    sid_ref = src.effective_source_id or "none"
    return f'  - [upload] "{label}" (source_id={sid_ref})'


def _format_request_context_block(request_context: Mapping[str, Any] | None) -> str:
    if not request_context:
        return ""

    lines: list[str] = []
    attached = parse_attached_sources(request_context.get("attached_sources"))
    if attached:
        lines.append("- Attached context sources:")
        for src in attached:
            lines.append(_format_attached_source_line(src))
        lines.append(
            "  Routing guidance: Use `rag_agent` for full-text content questions on uploads or "
            "library papers marked `full-text indexed` / `PDF auto-indexable`. Use `library_agent` "
            "(passing exact `item_id`) for citations/BibTeX, notes, metadata, tags, collections, "
            "organizing/trashing, or `metadata/notes only` papers. For combined tasks (e.g., summarize "
            "PDF content then save note/cite/tag in Library), dispatch `rag_agent` first with "
            '`"needs_followup": true`, then `library_agent` using the exact `item_id`.'
        )

    covered_ids = {
        value
        for src in attached
        for value in (src.effective_source_id, src.item_id, src.file_id)
        if value
    }
    doc_ids = request_context.get("document_ids")
    if isinstance(doc_ids, Sequence) and not isinstance(doc_ids, str | bytes) and doc_ids:
        extra_docs = [
            str(d).strip()
            for d in doc_ids
            if str(d).strip() and str(d).strip() not in covered_ids
        ]
        if extra_docs:
            joined = ", ".join(extra_docs)
            lines.append(
                f"- Attached workspace document_ids: [{joined}] "
                "(Use `rag_agent` when answering from or inspecting these uploaded workspace documents.)"
            )

    page_id = str(request_context.get("page_id") or "").strip()
    project_id = str(request_context.get("project_id") or "").strip()
    if project_id:
        lines.append(
            f"- Active project_id: {project_id} "
            "(Delegate project/work-item/Kanban/member/workload/activity requests to "
            "`project_agent`. When paired with attached papers or LaTeX tasks, coordinate "
            'with `rag_agent` / `editor_agent` via `"needs_followup": true`.)'
        )
    filename = str(
        request_context.get("active_file_path") or request_context.get("filename") or ""
    ).strip()
    has_file_content = bool(str(request_context.get("file_content") or "").strip())
    has_selection = bool(str(request_context.get("selection") or "").strip())
    has_compile_errors = bool(request_context.get("compile_errors"))
    if page_id:
        lines.append(f"- Active workspace page_id: {page_id}")
    if filename or has_file_content or has_selection or has_compile_errors or page_id:
        editor_cues: list[str] = []
        if filename:
            editor_cues.append(f"file={filename}")
        if request_context.get("cursor_line") is not None:
            editor_cues.append(f"cursor_line={request_context['cursor_line']}")
        if has_selection:
            editor_cues.append("has_selection=true")
        if has_compile_errors:
            editor_cues.append("has_compile_errors=true")
        cue_summary = ", ".join(editor_cues) if editor_cues else "active page"
        lines.append(
            f"- Active LaTeX editor context ({cue_summary}): "
            "Delegate LaTeX editing, academic polishing, error debugging, equation/section drafting, "
            "page explanation, citation-gap audits, multi-file structure/outline/cross-reference/"
            "package audits, collaborator review-comment handling, and `.tex`/`.bib` inspection to "
            "`editor_agent`. If the task also requires extracting evidence "
            "from attached PDFs (`rag_agent`) or finding papers/DOIs in Library/web (`library_agent` / "
            "`web_search_agent`) before citing or writing, dispatch those prerequisite agents first with "
            '`"needs_followup": true`, then delegate to `editor_agent`. Similarly, when the user asks to '
            "scan the page for unsupported claims and fetch missing references, run `editor_agent` "
            'with `"needs_followup": true` to identify gaps before dispatching `library_agent` / '
            "`web_search_agent`."
        )

    allowed = request_context.get("allowed_domains") or request_context.get("include_domains")
    blocked = request_context.get("blocked_domains") or request_context.get("exclude_domains")
    target_urls = request_context.get("target_urls") or request_context.get("urls")
    ext_sources = request_context.get("external_sources") or request_context.get("client_sources")
    search_mode = str(request_context.get("search_mode") or "").strip()

    ext_signals: list[str] = []
    if isinstance(allowed, Sequence) and not isinstance(allowed, str | bytes) and allowed:
        ext_signals.append(f"allowed_domains={list(allowed)}")
    if isinstance(blocked, Sequence) and not isinstance(blocked, str | bytes) and blocked:
        ext_signals.append(f"blocked_domains={list(blocked)}")
    if (
        isinstance(target_urls, Sequence)
        and not isinstance(target_urls, str | bytes)
        and target_urls
    ):
        ext_signals.append(f"target_urls={list(target_urls)}")
    if isinstance(ext_sources, Sequence) and not isinstance(ext_sources, str | bytes) and ext_sources:
        ext_signals.append(f"external_sources={len(ext_sources)}")
    if search_mode:
        ext_signals.append(f"search_mode={search_mode}")

    if ext_signals:
        lines.append(
            f"- External search/fetch scope ({'; '.join(ext_signals)}): "
            "External web/scholar scope or URLs are attached — delegate to `web_search_agent` "
            "rather than answering from memory."
        )

    ws_memory = request_context.get("workspace_memory")
    if isinstance(ws_memory, Mapping) and ws_memory:
        mem_parts = [f"{k}={dict(v)}" for k, v in list(ws_memory.items())[:5] if isinstance(v, Mapping)]
        if mem_parts:
            lines.append(f"- Persistent workspace preferences: {'; '.join(mem_parts)}")

    if not lines:
        return ""
    return "## Active Request Context\n" + "\n".join(lines) + "\n\n"


def build_supervisor_prompt(
    catalog: str,
    *,
    round_count: int = 1,
    max_rounds: int = 3,
    plan_summary: str = "",
    prior_outputs: Sequence[AgentOutput] = (),
    request_context: Mapping[str, Any] | None = None,
    context_envelope: ContextEnvelope | None = None,
) -> str:
    """Build the Supervisor prompt ordered Static-First, Volatile-Last for KV Prefix Caching."""
    static_prefix = (
        "You are the central Supervisor orchestrator. Analyze the user's request, active request "
        "context, recent conversation, and any completed specialist outputs, then decide the next step.\n\n"
        "## Registered Specialist Agents\n"
        f"{catalog}\n\n"
        "## Output Contract\n"
        "Do NOT invoke any native function or tool calls; return plain JSON text only.\n"
        "Output ONLY a valid JSON object immediately (no `<think>` blocks, no markdown fences, no extra text):\n"
        '{"mode": "direct" | "delegate" | "synthesize", '
        '"targets": [{"agent": "<agent_name>", "instruction": "<self-contained task>"}], '
        '"needs_followup": false, '
        '"plan_summary": "<concise multi-step plan if needs_followup is true>", '
        '"reasoning": "<brief reason>"}\n\n'
        "## Rules\n"
        '1. On round 1 with no specialist needed and no attached workspace/external cues requiring '
        'retrieval, choose "mode": "direct" (empty targets).\n'
        '2. For a single-step or independent parallel task, choose "mode": "delegate" with '
        '"needs_followup": false. When a question asks to compare or combine internal workspace '
        "documents (`rag_agent`) with external web/scholarly literature (`web_search_agent`), "
        'delegate to BOTH agents in parallel in "targets" with "needs_followup": false.\n'
        "3. For a sequential multi-step task where a later agent depends on an earlier agent's "
        'output, dispatch ONLY the prerequisite agent(s) now with "needs_followup": true and '
        'record the roadmap in "plan_summary".\n'
        '4. On follow-up rounds (round > 1), if Prior Specialist Outputs are sufficient, choose '
        '"mode": "synthesize" (empty targets); otherwise delegate the next dependent step.\n'
        "5. Only reference agent names listed in Registered Specialist Agents.\n"
        "6. Preserve the user's language in each target instruction."
    )

    volatile_parts: list[str] = []
    if context_envelope and context_envelope.primacy_text.strip():
        volatile_parts.append(context_envelope.primacy_text.strip())

    context_block = _format_request_context_block(request_context).strip()
    if context_block:
        volatile_parts.append(context_block)

    if context_envelope and context_envelope.middle_text.strip():
        volatile_parts.append(context_envelope.middle_text.strip())

    if prior_outputs:
        progress_lines: list[str] = [f"## Progress (Round {round_count}/{max_rounds})"]
        if plan_summary.strip():
            progress_lines.append(f"Plan Summary: {plan_summary.strip()}")
        progress_lines.append("Prior Specialist Outputs:")
        for out in prior_outputs:
            if out.status == "error":
                progress_lines.append(
                    f"- [{out.agent}] (ERROR): {(out.error or 'Unknown error')[:400]}"
                )
            else:
                snippet = out.content.strip()
                if len(snippet) > 600:
                    snippet = snippet[:600] + "...[truncated]"
                progress_lines.append(f"- [{out.agent}] (OK): {snippet}")
        volatile_parts.append("\n".join(progress_lines))

    if not volatile_parts:
        return static_prefix
    return static_prefix + "\n\n" + "\n\n".join(volatile_parts)


def build_synthesizer_prompt(outputs: Sequence[AgentOutput]) -> str:
    """Build the Synthesizer system prompt from specialist AgentOutput records."""
    sections: list[str] = []
    global_cite_idx = 0
    for out in outputs:
        if out.status == "error":
            sections.append(
                f"### Specialist: {out.agent} [ERROR]\nError: {out.error or 'Unknown error'}"
            )
            continue

        lines = [f"### Specialist: {out.agent}", out.content.strip() or "(empty output)"]
        if out.citations:
            citation_lines = []
            for cite in out.citations:
                global_cite_idx += 1
                parts = [f"{format_evidence_tag(global_cite_idx)} (id={cite.source_id})"]
                if cite.title:
                    parts.append(cite.title)
                if cite.page is not None:
                    parts.append(f"(p. {cite.page})")
                if cite.url:
                    parts.append(cite.url)
                citation_lines.append(" ".join(parts))
            lines.append("Citations:\n" + "\n".join(citation_lines))
        sections.append("\n".join(lines))

    findings = "\n\n---\n\n".join(sections) if sections else "(no specialist findings)"
    return (
        "You are the final response synthesizer. Combine the specialist findings below into a "
        "clear, coherent, accurate answer in the user's language.\n\n"
        "## Grounding Rules\n"
        "1. Rely strictly on the specialist findings below; do not invent facts, tool outcomes, "
        "or citations.\n"
        "2. If any specialist reported [ERROR], honestly acknowledge that unavailable part while "
        "presenting all successful findings accurately.\n"
        f"3. {CITATION_INSTRUCTIONS}\n\n"
        f"## Specialist Findings\n{findings}"
    )
