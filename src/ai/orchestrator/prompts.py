"""System prompts for the abstract Supervisor planner and Synthesizer."""

from collections.abc import Mapping, Sequence
from typing import Any

from ai.orchestrator.citations import CITATION_INSTRUCTIONS, format_evidence_tag
from ai.orchestrator.context_sections import render_request_context
from ai.orchestrator.contracts import AgentOutput

DIRECT_RESPOND_SYSTEM_PROMPT = (
    "You are Flux AI Assistant, a helpful, concise workspace assistant for Flux.\n"
    "Rules:\n"
    "1. Always respond strictly in the same language as the user's latest message "
    "(for example, Vietnamese if the user writes in Vietnamese). Never switch languages "
    "mid-reply.\n"
    "2. If asked about your identity, introduce yourself as Flux AI Assistant. "
    "Never claim to be DeepSeek, OpenAI, Gemini, Claude, or any other third-party model "
    "or vendor.\n"
    "3. Keep greetings and direct answers clear, natural, and concise."
)


def format_user_profile_line(request_context: Mapping[str, Any] | None) -> str:
    """Format signed-in user profile metadata if provided by the backend."""
    if not isinstance(request_context, Mapping):
        return ""
    raw = request_context.get("user_profile")
    if not isinstance(raw, Mapping):
        return ""
    name = str(raw.get("name") or "").strip()
    email = str(raw.get("email") or "").strip()
    institution = str(raw.get("institution") or "").strip()
    parts: list[str] = []
    if name:
        parts.append(f"Name: {name}")
    if email:
        parts.append(f"Email: {email}")
    if institution:
        parts.append(f"Institution: {institution}")
    if not parts:
        return ""
    return "Signed-in user profile: " + " | ".join(parts)


def build_direct_system_prompt(request_context: Mapping[str, Any] | None = None) -> str:
    """Build Static-First direct-response system prompt with optional signed-in user identity."""
    user_line = format_user_profile_line(request_context)
    if not user_line:
        return DIRECT_RESPOND_SYSTEM_PROMPT
    return f"{DIRECT_RESPOND_SYSTEM_PROMPT}\n\n{user_line}"


def build_supervisor_prompt(
    catalog: str,
    *,
    round_count: int = 1,
    max_rounds: int = 3,
    plan_summary: str = "",
    prior_outputs: Sequence[AgentOutput] = (),
    request_context: Mapping[str, Any] | None = None,
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
        '1. On round 1, choose "mode": "direct" (empty targets) whenever the user\'s latest '
        "request can be satisfied without reading attached/workspace documents, querying workspace "
        "state, or searching external sources — regardless of whether papers or files are "
        "currently open or attached.\n"
        '2. For a single-step or independent parallel task, choose "mode": "delegate" with '
        '"needs_followup": false. When a question asks to compare or combine internal workspace '
        "documents (`rag_agent`) with external web/scholarly literature (`web_search_agent`), "
        'delegate to BOTH agents in parallel in "targets" with "needs_followup": false.\n'
        "3. For a sequential multi-step task where a later agent depends on an earlier agent's "
        'output, dispatch ONLY the prerequisite agent(s) now with "needs_followup": true and '
        'record the roadmap in "plan_summary".\n'
        "4. On follow-up rounds (round > 1), if Prior Specialist Outputs are sufficient, choose "
        '"mode": "synthesize" (empty targets); otherwise delegate the next dependent step.\n'
        "5. Only reference agent names listed in Registered Specialist Agents.\n"
        "6. Preserve the user's language in each target instruction.\n"
        "7. Creating, initializing, writing, or editing a new Page/article/paper (e.g., `tạo bài viết`, "
        "`khởi tạo bài viết`, `tạo page`, `create page`) belongs to `editor_agent` — NEVER delegate "
        "article/page creation to `project_agent` as a Kanban work item.\n"
        "8. Do NOT invoke `web_search_agent` for questions strictly about an internal workspace "
        "project, its internal metrics, ADRs, postmortems, or team ownership unless the user "
        "explicitly asks for external public/web/scholarly sources."
    )

    volatile_parts: list[str] = []
    context_block = render_request_context(request_context).strip()
    if context_block:
        volatile_parts.append(context_block)

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
