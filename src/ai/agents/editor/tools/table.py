"""Tool for fetching experiment results from GitHub and synthesizing publication-grade LaTeX tables."""

from __future__ import annotations

import re
from functools import partial

from pydantic import BaseModel, Field

from ai.agents.editor.patching.contracts import PatchKind
from ai.agents.editor.patching.pipeline import build_verified_patch
from ai.agents.editor.tools.contracts import (
    EditorToolResult,
    EditorToolRuntime,
    EditorToolSpec,
)
from ai.agents.editor.tools.file_ops import resolve_and_cache_file
from ai.shared.context.contracts import EntityDomain, GraphNode, RelationType
from ai.shared.context.facade import ResearchContextFacade
from ai.shared.github.client import HttpGitHubArtifactClient
from ai.shared.github.contracts import (
    GitHubArtifactContent,
    GitHubArtifactError,
    GitHubArtifactGateway,
)
from ai.shared.tabular.builder import LatexBooktabsBuilder
from ai.shared.tabular.contracts import TableStyleOptions, TabularMatrix
from ai.shared.tabular.parser import TabularDataParser


class InsertExperimentTableArgs(BaseModel):
    """Arguments for fetching GitHub experiment outputs and proposing a LaTeX table patch."""

    repo: str = Field(
        ...,
        description="GitHub repository in 'owner/repo' format (e.g. 'deepmind/alpha-exp').",
    )
    file_path: str = Field(
        ...,
        description="Path to metric file in repo (e.g. 'eval/results.csv' or 'benchmark.json').",
    )
    ref: str = Field(
        default="main",
        description="Branch, tag, or 40-character commit SHA.",
    )
    caption: str = Field(
        default="Experimental Results",
        description="Scientific caption for the LaTeX table.",
    )
    label: str = Field(
        default="tab:results",
        description="LaTeX label for cross-referencing (e.g. 'tab:eval').",
    )
    target_file: str | None = Field(
        default=None,
        description="Target page file path (defaults to active file or 'main.tex').",
    )
    target_section: str | None = Field(
        default=None,
        description="Optional section title (e.g. 'Experiments', 'Evaluation') to insert after.",
    )
    search_text: str | None = Field(
        default=None,
        description="Exact 2-5 anchor lines to insert around if specified explicitly.",
    )
    precision: int = Field(
        default=2,
        ge=0,
        le=6,
        description="Decimal places for floating point metrics.",
    )


def _find_section_anchor(content: str, section_title: str) -> str | None:
    norm_title = section_title.strip().lower()
    pattern = re.compile(
        r"\\(?:sub)*section\*?\{([^}]+)\}",
        re.IGNORECASE,
    )
    for line in content.splitlines():
        match = pattern.search(line)
        if match and norm_title in match.group(1).strip().lower():
            return line
    return None


def _parse_matrix(artifact: GitHubArtifactContent, precision: int) -> TabularMatrix:
    parser = TabularDataParser(precision=precision)
    if artifact.format == "json":
        return parser.parse_json(artifact.content)
    return parser.parse_csv(artifact.content, delimiter="\t" if artifact.format == "tsv" else ",")


def _render_table(
    artifact: GitHubArtifactContent,
    matrix: TabularMatrix,
    args: InsertExperimentTableArgs,
) -> str:
    notes = (
        f"Generated from GitHub repository \\texttt{{{artifact.repo}}} "
        f"at commit \\texttt{{{artifact.commit_sha[:7]}}}."
    )
    options = TableStyleOptions(
        caption=args.caption, label=args.label, precision=args.precision, notes=notes
    )
    return LatexBooktabsBuilder(matrix=matrix, options=options).build()


def _choose_anchor(content: str, args: InsertExperimentTableArgs) -> tuple[str, PatchKind]:
    """Pick where the table goes: explicit anchor, target section, `\\end{document}`, or EOF."""
    explicit = (args.search_text or "").strip()
    if explicit:
        return explicit, "insert_after"
    if args.target_section:
        section_line = _find_section_anchor(content, args.target_section)
        if section_line:
            return section_line, "insert_after"
    if "\\end{document}" in content:
        return "\\end{document}", "insert_before"
    last_lines = [line for line in content.splitlines() if line.strip()]
    return (last_lines[-1] if last_lines else ""), "insert_after"


def _record_provenance(
    facade: ResearchContextFacade,
    *,
    space_id: str,
    file_path: str,
    args: InsertExperimentTableArgs,
    artifact: GitHubArtifactContent,
) -> None:
    table_node = GraphNode(
        id=f"page:{space_id}#{args.label}",
        domain=EntityDomain.PAGE,
        label=args.caption,
        attributes={"file_path": file_path, "label": args.label, "has_benchmark": True},
    )
    artifact_node = GraphNode(
        id=f"github:{artifact.repo}@{artifact.commit_sha}:{artifact.path}",
        domain=EntityDomain.EXPERIMENT,
        label=f"{artifact.repo}/{artifact.path}",
        attributes={
            "repo": artifact.repo,
            "commit_sha": artifact.commit_sha,
            "url": artifact.url,
        },
    )
    facade.register_node(table_node)
    facade.register_node(artifact_node)
    facade.link_entities(
        src_id=artifact_node.id,
        dst_id=table_node.id,
        relation=RelationType.PROVES,
        metadata={"path": artifact.path},
    )


def _failure(error: str) -> EditorToolResult:
    return EditorToolResult(tool="insert_experiment_table", status="error", error=error)


async def handle_insert_experiment_table(
    runtime: EditorToolRuntime,
    args: InsertExperimentTableArgs,
    *,
    artifact_client: GitHubArtifactGateway | None = None,
    context_facade: ResearchContextFacade | None = None,
) -> EditorToolResult:
    """Fetch experiment data from GitHub, synthesize LaTeX booktabs table, and propose patch."""
    if len(runtime.patches) >= runtime.max_patches:
        return _failure(f"Reached maximum patch proposals ({runtime.max_patches}) for one turn.")

    client = artifact_client or HttpGitHubArtifactClient()
    try:
        artifact = await client.fetch_artifact(repo=args.repo, path=args.file_path, ref=args.ref)
    except GitHubArtifactError as exc:
        return _failure(f"Failed to fetch GitHub artifact: {exc.message}")

    matrix = _parse_matrix(artifact, args.precision)
    if not matrix.rows or not matrix.headers:
        return _failure(f"Failed to parse any tabular data from '{args.file_path}'.")
    table_latex = _render_table(artifact, matrix, args)

    target_path = (args.target_file or runtime.scope.active_file_path or "main.tex").lstrip("/")
    cached, err = await resolve_and_cache_file(runtime, file_path=target_path)
    if err or cached is None:
        return _failure(err or f"Could not resolve target file '{target_path}'.")

    search_text, kind = _choose_anchor(cached.content, args)
    outcome = build_verified_patch(
        runtime.patches,
        content=cached.content,
        file_id=cached.file_id,
        file_path=cached.file_path,
        kind=kind,
        search_text=search_text,
        replace_text="\n" + table_latex + "\n",
        title=f"Insert experiment table from {artifact.repo} ({args.label})",
        rationale=(
            f"Automated table synthesis from {artifact.repo}/{artifact.path} "
            f"at commit {artifact.commit_sha[:7]}"
        ),
        known_cite_keys=frozenset(runtime.known_cite_keys),
        known_labels=frozenset(runtime.known_labels),
    )
    if outcome.patch is None:
        return _failure(outcome.error or "Failed to anchor table insertion into page.")

    runtime.put_cached_file(
        file_id=cached.file_id, file_path=cached.file_path, content=outcome.patched_content
    )
    runtime.patches.append(outcome.patch)
    if context_facade is not None:
        _record_provenance(
            context_facade,
            space_id=runtime.scope.space_id,
            file_path=cached.file_path,
            args=args,
            artifact=artifact,
        )

    artifact_info = {
        "repo": artifact.repo,
        "path": artifact.path,
        "commit_sha": artifact.commit_sha,
        "url": artifact.url,
    }
    return EditorToolResult(
        tool="insert_experiment_table",
        status="ok",
        patch=outcome.patch,
        data={
            "table_latex": table_latex,
            "artifact": artifact_info,
            "provenance": {
                "src_id": f"page:{runtime.scope.space_id}#{args.label}",
                "dst_id": f"github:{artifact.repo}@{artifact.commit_sha}:{artifact.path}",
                "relation": "derived_from",
            },
            "rows_count": len(matrix.rows),
            "metrics_count": len(matrix.headers) - 1,
        },
    )


def build_table_tools(
    *,
    artifact_client: GitHubArtifactGateway | None = None,
    context_facade: ResearchContextFacade | None = None,
) -> tuple[EditorToolSpec, ...]:
    """Assemble table tools for the Editor Agent."""
    handler = (
        partial(
            handle_insert_experiment_table,
            artifact_client=artifact_client,
            context_facade=context_facade,
        )
        if artifact_client is not None or context_facade is not None
        else handle_insert_experiment_table
    )
    return (
        EditorToolSpec(
            name="insert_experiment_table",
            description=(
                "Fetch experiment results from a GitHub repository (CSV/JSON/TSV), format "
                "into a publication-grade LaTeX booktabs table (auto-bolding best values, "
                "calculating Mean ± Std), and propose a patch to insert it into the page."
            ),
            args_schema=InsertExperimentTableArgs,
            risk_tier="propose",
            idempotent=True,
            handler=handler,
        ),
    )
