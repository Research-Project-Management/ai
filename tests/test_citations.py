"""Unit tests for canonical inline citations, guards, and bibliography demotion."""

from langchain_core.runnables import RunnableConfig

from ai.agents.rag.retrieval.contracts import RankedPassage
from ai.agents.rag.retrieval.reranker import LexicalHybridReranker
from ai.agents.rag.workflow.prompts import build_grounded_answer_prompt, format_evidence_block
from ai.agents.web_search.workflow.prompts import build_web_synthesis_prompt
from ai.orchestrator.citations import (
    extract_cited_indices,
    sanitize_citation_markers,
    select_indexed_citations,
)
from ai.orchestrator.contracts import AgentOutput, CitationItem, DraftOutput
from ai.orchestrator.guards import CitationIntegrityGuard
from ai.orchestrator.prompts import build_synthesizer_prompt
from ai.shared.contracts.knowledge import KnowledgeHit


def test_extract_cited_indices_prefers_canonical_over_paper_brackets() -> None:
    text = (
        "ResNet-56 beats Maxout [9], NIN [25], and FitNet [34] on CIFAR-10 [[cite:6]]. "
        "Earlier experiments [[cite:1, 2]] and out-of-range [[cite:99]] are handled cleanly."
    )
    assert extract_cited_indices(text, max_index=6) == [6, 1, 2]


def test_extract_cited_indices_falls_back_to_legacy_brackets_when_no_canonical() -> None:
    text = "The resonator reaches 97.2% fidelity at 4K [1] and scales cleanly [3]."
    assert extract_cited_indices(text, max_index=4) == [1, 3]


def test_select_indexed_citations_preserves_canonical_sparse_index() -> None:
    candidates = [
        CitationItem(source_id="doc-1", title="ResNet", page=i, snippet=f"Snippet {i}")
        for i in range(1, 7)
    ]
    selected = select_indexed_citations(
        "ResNet-110 achieves 6.43% error [[cite:6]] while beating Maxout [9].",
        candidates,
    )
    assert len(selected) == 1
    assert selected[0].index == 6
    assert selected[0].page == 6
    assert selected[0].snippet == "Snippet 6"


async def test_citation_integrity_guard_preserves_distinct_indices_and_dedups_unindexed() -> None:
    guard = CitationIntegrityGuard()
    draft = DraftOutput(
        content="Claim A [[cite:2]] and Claim B [[cite:6]].",
        citations=(
            CitationItem(index=2, source_id="doc-1", title="ResNet", page=6, snippet="Same snippet"),
            CitationItem(index=6, source_id="doc-1", title="ResNet", page=6, snippet="Same snippet"),
            CitationItem(source_id="doc-2", title="Unindexed", page=1, snippet="Dup"),
            CitationItem(source_id="doc-2", title="Unindexed", page=1, snippet="Dup"),
            CitationItem(source_id="   ", title="Invalid"),
        ),
    )
    validated = await guard.validate(draft, state={}, config=RunnableConfig())
    assert [(c.index, c.source_id) for c in validated.citations] == [
        (2, "doc-1"),
        (6, "doc-1"),
        (None, "doc-2"),
    ]


async def test_reranker_soft_demotes_bibliography_unless_user_asks_for_references() -> None:
    reranker = LexicalHybridReranker()
    refs_hit = KnowledgeHit(
        source_id="paper-1",
        source_type="file",
        title="ResNet Paper",
        content=(
            "[1] K. Simonyan and A. Zisserman. Very deep convolutional networks.\n"
            "[2] C. Szegedy et al. Going deeper with convolutions.\n"
            "[3] S. Ioffe and C. Szegedy. Batch normalization."
        ),
        score=0.95,
        chunk_index=0,
        page=9,
        metadata={"section": "References"},
    )
    method_hit = KnowledgeHit(
        source_id="paper-1",
        source_type="file",
        title="ResNet Paper",
        content="Deep residual networks use identity shortcut connections for image recognition experiments.",
        score=0.80,
        chunk_index=1,
        page=3,
        metadata={"section": "3. Deep Residual Learning"},
    )

    summary_ranked = await reranker.rerank(
        "Summarize the paper method and experiments",
        [refs_hit, method_hit],
        top_k=2,
    )
    assert summary_ranked[0].chunk_index == 1

    ref_query_ranked = await reranker.rerank(
        "Liệt kê tài liệu tham khảo (references) trong bài báo",
        [refs_hit, method_hit],
        top_k=2,
    )
    assert ref_query_ranked[0].chunk_index == 0


def test_all_synthesis_prompts_enforce_canonical_cites_and_no_leak() -> None:
    rag_prompt = build_grounded_answer_prompt()
    web_prompt = build_web_synthesis_prompt()
    synth_prompt = build_synthesizer_prompt(
        [
            AgentOutput(
                agent="rag_agent",
                content="Summary [[cite:1]]",
                citations=(CitationItem(index=1, source_id="doc-1", title="ResNet", page=6),),
            )
        ]
    )

    for prompt in (rag_prompt, web_prompt, synth_prompt):
        assert "[[cite:N]]" in prompt
        assert "Never mention internal passage, chunk, or reference numbers" in prompt

    evidence = format_evidence_block(
        query="Summarize",
        passages_repacked=[
            RankedPassage(
                source_id="doc-1",
                source_type="file",
                title="ResNet",
                content="Residual blocks ease training.",
                rerank_score=0.9,
                retrieval_score=0.8,
                page=2,
            )
        ],
        ranked_desc=[
            RankedPassage(
                source_id="doc-1",
                source_type="file",
                title="ResNet",
                content="Residual blocks ease training.",
                rerank_score=0.9,
                retrieval_score=0.8,
                page=2,
            )
        ],
    )
    assert "[cite:1] Source: ResNet | Page 2" in evidence
    assert "[cite:1] (id=doc-1)" in synth_prompt


async def test_citation_integrity_guard_strips_phantom_citations() -> None:
    cleaned, valid_cnt, phantom_cnt = sanitize_citation_markers(
        "FluxKV cuts VRAM by 63.8% [[cite:1, 2]] and scales [[cite:99]].",
        valid_indices=frozenset({1, 2}),
    )
    assert cleaned == "FluxKV cuts VRAM by 63.8% [[cite:1]][[cite:2]] and scales."
    assert valid_cnt == 2
    assert phantom_cnt == 1

    guard = CitationIntegrityGuard()
    draft = DraftOutput(
        content="FluxKV reduces memory by 63.8% [[cite:1]] while dropping missing ref [[cite:8]].",
        citations=(
            CitationItem(
                index=1,
                source_id="doc-fluxkv",
                title="FluxKV 2026",
                page=3,
                snippet="Experiments confirm FluxKV reduces memory by 63.8% across benchmarks.",
            ),
        ),
    )

    validated = await guard.validate(draft, state={}, config=RunnableConfig())
    assert "[[cite:8]]" not in validated.content
    assert "[[cite:1]]" in validated.content
