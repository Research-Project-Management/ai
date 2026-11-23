"""Subgraph compiler and Root Orchestrator AgentSpec adapter for `rag_agent`."""

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ai.agents.rag.ingestion.lazy_indexer import LazyPdfIndexer
from ai.agents.rag.retrieval.engine import RetrievalEngine
from ai.agents.rag.workflow.contracts import RagAgentConfig
from ai.agents.rag.workflow.nodes import build_rag_nodes
from ai.agents.rag.workflow.state import RagState
from ai.orchestrator.contracts import AgentOutput, AgentSpec
from ai.orchestrator.state import RootState
from ai.shared.contracts.run_control import RunUsage
from ai.shared.contracts.sources import collect_rag_source_ids, parse_attached_sources
from ai.shared.llm.factory import ChatRunnable
from ai.shared.workflow.turn import latest_query, recent_tail


def build_rag_subgraph(
    llm: ChatRunnable,
    retrieval_engine: RetrievalEngine,
    *,
    config: RagAgentConfig | None = None,
) -> CompiledStateGraph:
    """Compile the 4-node Corrective Hybrid RAG StateGraph."""
    cfg = config or RagAgentConfig()
    nodes = build_rag_nodes(llm=llm, retrieval_engine=retrieval_engine, agent_config=cfg)

    def _route_after_eval(state: RagState) -> str:
        ranked = state.get("ranked_desc") or []
        if len(ranked) >= cfg.min_passages:
            return "answer"
        if state.get("attempt", 1) < cfg.max_attempts and not state.get("abstain", False):
            return "retry"
        return "answer"

    builder = StateGraph(RagState)
    builder.add_node("transform_query", nodes["transform_query"])
    builder.add_node("retrieve_passages", nodes["retrieve_passages"])
    builder.add_node("evaluate_crag", nodes["evaluate_crag"])
    builder.add_node("generate_answer", nodes["generate_answer"])

    builder.add_edge(START, "transform_query")
    builder.add_edge("transform_query", "retrieve_passages")
    builder.add_edge("retrieve_passages", "evaluate_crag")
    builder.add_conditional_edges(
        "evaluate_crag",
        _route_after_eval,
        {"retry": "transform_query", "answer": "generate_answer"},
    )
    builder.add_edge("generate_answer", END)
    return builder.compile()


def build_rag_agent_spec(
    llm: ChatRunnable,
    retrieval_engine: RetrievalEngine,
    *,
    config: RagAgentConfig | None = None,
    lazy_indexer: LazyPdfIndexer | None = None,
) -> AgentSpec:
    """Build the pluggable `rag_agent` AgentSpec backed by its isolated CRAG subgraph."""
    cfg = config or RagAgentConfig()
    subgraph = build_rag_subgraph(llm, retrieval_engine, config=cfg)

    async def _run_rag_agent(state: RootState, run_config: RunnableConfig) -> AgentOutput:
        configurable: dict[str, Any] = dict(run_config.get("configurable") or {})
        context: dict[str, Any] = dict(state.get("context") or {})

        space_id = str(configurable.get("space_id") or context.get("space_id") or "").strip()
        if not space_id:
            raise ValueError(
                "rag_agent requires a non-empty space_id in RunnableConfig.configurable"
            )

        raw_doc_ids = tuple(
            str(item).strip()
            for item in (configurable.get("document_ids") or context.get("document_ids") or ())
            if str(item).strip()
        )
        attached = parse_attached_sources(context.get("attached_sources"))
        lazy_indexed_count = 0
        lazy_report_artifacts: dict[str, Any] = {}

        if attached:
            if lazy_indexer is not None:
                report = await lazy_indexer.ensure_indexed(space_id=space_id, sources=attached)
                lazy_indexed_count = len(report.newly_indexed)
                metadata_only_items = {
                    s.item_id for s in attached if s.item_id and not s.effective_source_id
                }
                failed_ids = {f["source_id"] for f in report.failed if f.get("source_id")}
                extra_fallbacks = tuple(
                    doc_id
                    for doc_id in raw_doc_ids
                    if doc_id not in metadata_only_items
                    and doc_id not in failed_ids
                    and doc_id not in report.ready_source_ids
                )
                source_ids = (*report.ready_source_ids, *extra_fallbacks)
                if report.skipped_metadata_only or report.failed or lazy_indexed_count:
                    lazy_report_artifacts = {
                        "lazy_indexed_count": lazy_indexed_count,
                        "skipped_metadata_only": list(report.skipped_metadata_only),
                        "lazy_index_failed": list(report.failed),
                    }
            else:
                source_ids = collect_rag_source_ids(attached, raw_doc_ids)

            if not source_ids:
                return AgentOutput(
                    agent="rag_agent",
                    content=(
                        "Các tài liệu được đính kèm hiện chưa có toàn văn PDF được index trong kho tri thức "
                        "(hoặc chỉ có metadata/ghi chú). Hãy thử hỏi về thông tin thư mục, trích dẫn hoặc ghi chú."
                    ),
                    status="ok",
                    artifacts={"abstained": True, **lazy_report_artifacts},
                    is_final=True,
                )
        else:
            source_ids = raw_doc_ids

        raw_source_types = context.get("source_types") or ()
        source_types = tuple(str(item).strip() for item in raw_source_types if str(item).strip())

        query = latest_query(state)
        if not query:
            return AgentOutput(
                agent="rag_agent",
                content="Câu hỏi trống, không thể truy xuất tài liệu.",
                status="error",
                error="empty query",
            )

        sub_input: RagState = {
            "query": query,
            "conversation_tail": recent_tail(list(state.get("messages") or ())),
            "space_id": space_id,
            "source_ids": source_ids,
            "source_types": source_types,
            "attempt": 0,
        }
        result = await subgraph.ainvoke(sub_input, config=run_config)
        raw_usage = result.get("usage")
        return AgentOutput(
            agent="rag_agent",
            content=str(result.get("final_answer") or ""),
            status="ok",
            citations=tuple(result.get("citations") or ()),
            artifacts={
                "abstained": bool(result.get("abstain", False)),
                "attempts": int(result.get("attempt", 1)),
                "total_candidates": int(result.get("total_candidates", 0)),
                **lazy_report_artifacts,
            },
            usage=raw_usage if isinstance(raw_usage, RunUsage) else RunUsage(),
            is_final=True,
        )

    return AgentSpec(
        name="rag_agent",
        description=(
            "Searches and answers questions grounded in uploaded documents, PDFs, DOCX files, "
            "and Markdown notes inside the current workspace with inline citations."
        ),
        handler=_run_rag_agent,
        capabilities=("hybrid_search", "rerank", "crag", "grounded_citations"),
        timeout_seconds=cfg.timeout_seconds,
    )
