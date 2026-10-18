"""Unit tests for the Experiment-to-Paper (E2P) tabular synthesis and provenance pipeline."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from ai.agents.editor.gateway.contracts import EditorDocument, EditorFileInfo, EditorScope
from ai.agents.editor.tools.contracts import EditorToolRuntime
from ai.agents.editor.tools.registry import build_default_editor_tool_registry
from ai.agents.editor.tools.table import (
    InsertExperimentTableArgs,
    build_table_tools,
    handle_insert_experiment_table,
)
from ai.shared.context.contracts import EntityDomain, RelationType
from ai.shared.context.facade import ResearchContextFacade
from ai.shared.github.client import HttpGitHubArtifactClient
from ai.shared.github.contracts import GitHubArtifactContent, GitHubArtifactError
from ai.shared.tabular.builder import LatexBooktabsBuilder
from ai.shared.tabular.contracts import MetricDirection, TableStyleOptions
from ai.shared.tabular.heuristics import infer_metric_direction, is_proposed_method_name
from ai.shared.tabular.parser import TabularDataParser


def test_heuristic_metric_polarities_and_proposed_methods() -> None:
    # Maximization metrics
    assert infer_metric_direction("Accuracy (%)") == MetricDirection.MAXIMIZE
    assert infer_metric_direction("Top-1 Acc") == MetricDirection.MAXIMIZE
    assert infer_metric_direction("F1-Score") == MetricDirection.MAXIMIZE
    assert infer_metric_direction("mAP@50") == MetricDirection.MAXIMIZE
    assert infer_metric_direction("BLEU") == MetricDirection.MAXIMIZE

    # Minimization metrics
    assert infer_metric_direction("Cross-Entropy Loss") == MetricDirection.MINIMIZE
    assert infer_metric_direction("Perplexity (PPL)") == MetricDirection.MINIMIZE
    assert infer_metric_direction("Latency (ms)") == MetricDirection.MINIMIZE
    assert infer_metric_direction("Word Error Rate (WER)") == MetricDirection.MINIMIZE
    assert infer_metric_direction("RMSE") == MetricDirection.MINIMIZE

    # Neutral metrics
    assert infer_metric_direction("Sample Count") == MetricDirection.NEUTRAL

    # Proposed method detection
    assert is_proposed_method_name("Ours (Flux)") is True
    assert is_proposed_method_name("Proposed Architecture") is True
    assert is_proposed_method_name("ResNet-50 Baseline") is False


def test_tabular_parser_single_run_and_highlighting() -> None:
    csv_data = """Model,Accuracy,Loss,Latency_ms
ResNet-50,76.2,0.85,15.2
Transformer-B,81.4,0.62,22.1
Ours (Flux),84.6,0.48,12.0
"""
    parser = TabularDataParser(precision=2)
    matrix = parser.parse_csv(csv_data)

    assert matrix.headers == ("Model", "Accuracy", "Loss", "Latency_ms")
    assert len(matrix.rows) == 3

    # Ours (Flux) should have best accuracy (84.6), best loss (0.48), best latency (12.0)
    ours_row = next(r for r in matrix.rows if r.method == "Ours (Flux)")
    assert ours_row.is_proposed_method is True
    assert ours_row.cells["Accuracy"].is_best is True
    assert ours_row.cells["Loss"].is_best is True
    assert ours_row.cells["Latency_ms"].is_best is True

    # Transformer-B should have second best accuracy (81.4) and second best loss (0.62)
    tb_row = next(r for r in matrix.rows if r.method == "Transformer-B")
    assert tb_row.cells["Accuracy"].is_second_best is True
    assert tb_row.cells["Loss"].is_second_best is True


def test_tabular_parser_multi_seed_aggregation() -> None:
    # 3 seeds for each method
    csv_data = """Method,Acc,Loss
Baseline,80.0,0.50
Baseline,82.0,0.52
Baseline,81.0,0.48
Ours,88.0,0.30
Ours,89.0,0.32
Ours,90.0,0.28
"""
    parser = TabularDataParser(precision=2)
    matrix = parser.parse_csv(csv_data)

    assert len(matrix.rows) == 2
    baseline = next(r for r in matrix.rows if r.method == "Baseline")
    ours = next(r for r in matrix.rows if r.method == "Ours")

    # Baseline: mean Acc = 81.0, std = 1.0
    assert baseline.cells["Acc"].value == 81.0
    assert baseline.cells["Acc"].std == 1.0
    assert "81.00 $\\pm$ 1.00" in baseline.cells["Acc"].display_text

    # Ours: mean Acc = 89.0, std = 1.0 (Best)
    assert ours.cells["Acc"].is_best is True
    assert "89.00 $\\pm$ 1.00" in ours.cells["Acc"].display_text


def test_tabular_parser_json_format() -> None:
    json_data = json.dumps(
        [
            {"model": "BERT-Base", "f1": 88.5, "loss": 0.35},
            {"model": "RoBERTa", "f1": 90.2, "loss": 0.28},
            {"model": "Flux-Encoder", "f1": 92.4, "loss": 0.22},
        ]
    )
    parser = TabularDataParser(precision=1)
    matrix = parser.parse_json(json_data)

    assert matrix.headers == ("model", "f1", "loss")
    assert len(matrix.rows) == 3
    flux_row = next(r for r in matrix.rows if r.method == "Flux-Encoder")
    assert flux_row.cells["f1"].is_best is True
    assert flux_row.cells["loss"].is_best is True


def test_latex_booktabs_builder_rendering() -> None:
    csv_data = """Model,Accuracy,Loss
ResNet-50,76.2,0.85
Ours,84.6,0.48
"""
    matrix = TabularDataParser(precision=2).parse_csv(csv_data)
    builder = LatexBooktabsBuilder(
        matrix=matrix,
        options=TableStyleOptions(
            caption="ImageNet Evaluation",
            label="tab:imagenet",
            position="t",
            booktabs=True,
            notes="Evaluated on test set.",
        ),
    )
    latex = builder.build()

    assert "\\begin{table}[t]" in latex
    assert "\\centering" in latex
    assert "\\small" in latex
    assert "\\caption{ImageNet Evaluation}" in latex
    assert "\\label{tab:imagenet}" in latex
    assert "\\toprule" in latex
    assert "\\midrule" in latex
    assert "\\bottomrule" in latex
    assert "\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Loss}" in latex
    assert "\\textbf{Ours}" in latex
    assert "\\textbf{84.60}" in latex
    assert "\\end{tabular}" in latex
    assert "\\end{table}" in latex
    assert "Evaluated on test set." in latex


async def test_http_github_artifact_client_mock() -> None:
    sample_csv = "model,acc\nFlux,95.0\nBaseline,88.0\n"
    encoded = base64.b64encode(sample_csv.encode("utf-8")).decode("utf-8")

    def _mock_handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/repos/org/repo/contents/results.csv":
            return httpx.Response(
                200,
                json={
                    "name": "results.csv",
                    "path": "results.csv",
                    "sha": "blob-sha-123",
                    "size": len(sample_csv),
                    "encoding": "base64",
                    "content": encoded,
                },
            )
        if req.url.path == "/repos/org/repo/commits/main":
            return httpx.Response(
                200,
                json={"sha": "40charsha00000000000000000000000000000000"},
            )
        if "forbidden" in req.url.path:
            return httpx.Response(403, json={"message": "Forbidden"})
        return httpx.Response(404, json={"message": "Not Found"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(_mock_handler)) as client:
        gh_client = HttpGitHubArtifactClient(http_client=client)

        res = await gh_client.fetch_artifact(repo="org/repo", path="results.csv", ref="main")
        assert res.repo == "org/repo"
        assert res.format == "csv"
        assert "Flux,95.0" in res.content
        assert res.commit_sha == "40charsha00000000000000000000000000000000"

        # 404 error check
        with pytest.raises(GitHubArtifactError) as exc_404:
            await gh_client.fetch_artifact(repo="org/repo", path="missing.csv")
        assert exc_404.value.status_code == 404

        # 403 error check
        with pytest.raises(GitHubArtifactError) as exc_403:
            await gh_client.fetch_artifact(repo="org/forbidden", path="secret.csv")
        assert exc_403.value.status_code == 403


class _InMemoryFakeEditorGateway:
    def __init__(self, initial_doc: str) -> None:
        self.doc = initial_doc

    async def list_files(
        self, scope: EditorScope, *, include_content: bool = False
    ) -> list[EditorFileInfo]:
        del scope, include_content
        return [EditorFileInfo(id="file-1", path="main.tex", content=self.doc)]

    async def read_file(self, scope: EditorScope, doc_id: str) -> EditorDocument:
        del scope, doc_id
        return EditorDocument(id="file-1", path="main.tex", content=self.doc)


class _FakeGitHubClient:
    async def fetch_artifact(
        self,
        *,
        repo: str,
        path: str,
        ref: str = "main",
        auth_token: str | None = None,
    ) -> GitHubArtifactContent:
        del auth_token
        csv_text = "Model,F1,Loss\nBaseline,82.0,0.50\nOurs,89.5,0.31\n"
        return GitHubArtifactContent(
            repo=repo,
            path=path,
            ref=ref,
            commit_sha="a8f2c3e4b5d6e7f8a8f2c3e4b5d6e7f8a8f2c3e4",
            content=csv_text,
            format="csv",
            size_bytes=len(csv_text),
            url=f"https://github.com/{repo}/blob/main/{path}",
        )


async def test_editor_insert_experiment_table_tool() -> None:
    doc = """\\documentclass{article}
\\begin{document}
\\section{Introduction}
Some intro text.

\\section{Experiments}
We evaluate our method here.

\\end{document}
"""
    gateway = _InMemoryFakeEditorGateway(doc)
    runtime = EditorToolRuntime(
        gateway=gateway,  # type: ignore[arg-type]
        scope=EditorScope(space_id="space-1", active_file_path="main.tex"),
    )
    fake_gh = _FakeGitHubClient()

    args = InsertExperimentTableArgs(
        repo="flux-lab/vision-exp",
        file_path="eval/metrics.csv",
        caption="Benchmark Performance",
        label="tab:vision_eval",
        target_section="Experiments",
        precision=2,
    )

    result = await handle_insert_experiment_table(
        runtime,
        args,
        artifact_client=fake_gh,
    )

    assert result.status == "ok"
    assert result.patch is not None
    assert len(runtime.patches) == 1

    # Verify patched content has the table
    patched_file = runtime.get_cached_file(file_path="main.tex")
    assert patched_file is not None
    assert "\\begin{table}" in patched_file.content
    assert "\\caption{Benchmark Performance}" in patched_file.content
    assert "\\label{tab:vision_eval}" in patched_file.content
    assert "\\textbf{89.50}" in patched_file.content
    assert "\\bottomrule" in patched_file.content

    # Verify provenance data
    data = result.data or {}
    assert data["rows_count"] == 2
    assert data["metrics_count"] == 2
    assert data["artifact"]["repo"] == "flux-lab/vision-exp"
    assert data["artifact"]["commit_sha"] == "a8f2c3e4b5d6e7f8a8f2c3e4b5d6e7f8a8f2c3e4"
    assert data["provenance"]["relation"] == "derived_from"


def test_build_table_tools_registry() -> None:
    tools = build_table_tools()
    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "insert_experiment_table"
    assert tool.risk_tier == "propose"


async def test_editor_insert_experiment_table_registers_provenance_lineage() -> None:
    doc = """\\documentclass{article}
\\begin{document}
\\section{Experiments}
\\end{document}
"""
    gateway = _InMemoryFakeEditorGateway(doc)
    runtime = EditorToolRuntime(
        gateway=gateway,  # type: ignore[arg-type]
        scope=EditorScope(space_id="space-1", active_file_path="main.tex"),
    )
    fake_gh = _FakeGitHubClient()
    facade = ResearchContextFacade()

    registry = build_default_editor_tool_registry(
        artifact_client=fake_gh,
        context_facade=facade,
    )
    spec = registry.get("insert_experiment_table")
    assert spec is not None

    args = InsertExperimentTableArgs(
        repo="flux-lab/vision-exp",
        file_path="eval/metrics.csv",
        caption="Vision Benchmark",
        label="tab:vision_eval",
        target_section="Experiments",
    )
    result = await spec.handler(runtime, args)
    assert result.status == "ok"

    table_id = "page:space-1#tab:vision_eval"
    artifact_id = "github:flux-lab/vision-exp@a8f2c3e4b5d6e7f8a8f2c3e4b5d6e7f8a8f2c3e4:eval/metrics.csv"

    table_node = facade.graph.get_node(table_id)
    artifact_node = facade.graph.get_node(artifact_id)
    assert table_node is not None and table_node.domain == EntityDomain.PAGE
    assert artifact_node is not None and artifact_node.domain == EntityDomain.EXPERIMENT
    assert facade.graph.has_edge(artifact_id, table_id, RelationType.PROVES) is True

    lineage = facade.trace_claim_origin(table_id)
    assert len(lineage) == 1
    assert lineage[0].id == artifact_id

    audit_001 = [
        f
        for f in facade.audit_integrity()
        if f.rule_id == "AUDIT_001" and table_id in f.entity_ids
    ]
    assert audit_001 == []


def test_editor_graph_extracts_provenance_artifacts_from_observations() -> None:
    from ai.agents.editor.workflow.graph import _extract_provenance_artifacts

    observations = [
        {"tool": "read_file", "status": "ok", "data": {"content": "..."}},
        {
            "tool": "insert_experiment_table",
            "status": "ok",
            "data": {
                "artifact": {
                    "repo": "flux-lab/vision-exp",
                    "path": "eval/metrics.csv",
                    "commit_sha": "a8f2c3e4b5d6e7f8a8f2c3e4b5d6e7f8a8f2c3e4",
                    "url": "https://github.com/flux-lab/vision-exp/blob/a8f2c3e/eval/metrics.csv",
                }
            },
        },
        # Duplicate entry is deduplicated
        {
            "tool": "insert_experiment_table",
            "status": "ok",
            "data": {
                "artifact": {
                    "repo": "flux-lab/vision-exp",
                    "path": "eval/metrics.csv",
                    "commit_sha": "a8f2c3e4b5d6e7f8a8f2c3e4b5d6e7f8a8f2c3e4",
                    "url": "https://github.com/flux-lab/vision-exp/blob/a8f2c3e/eval/metrics.csv",
                }
            },
        },
        {"tool": "insert_experiment_table", "status": "error", "error": "404"},
    ]

    extracted = _extract_provenance_artifacts(observations)
    assert extracted == [
        {
            "repo": "flux-lab/vision-exp",
            "path": "eval/metrics.csv",
            "commit_sha": "a8f2c3e4b5d6e7f8a8f2c3e4b5d6e7f8a8f2c3e4",
            "url": "https://github.com/flux-lab/vision-exp/blob/a8f2c3e/eval/metrics.csv",
        }
    ]
