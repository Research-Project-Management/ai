"""Workflow nodes for query transformation, retrieval, CRAG evaluation, and grounded answering."""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ai.agents.rag.retrieval.contracts import RetrievalQuery
from ai.agents.rag.retrieval.engine import RetrievalEngine
from ai.agents.rag.workflow.contracts import (
    ABSTENTION_RESPONSE,
    QueryTransformResult,
    RagAgentConfig,
)
from ai.agents.rag.workflow.prompts import (
    build_grounded_answer_prompt,
    build_query_transform_prompt,
    format_evidence_block,
)
from ai.agents.rag.workflow.state import RagState
from ai.orchestrator.citations import select_indexed_citations
from ai.shared.contracts.run_control import RunUsage
from ai.shared.llm.factory import ChatRunnable
from ai.shared.llm.nostream import with_nostream
from ai.shared.llm.output import extract_text, parse_json_object, strip_reasoning


def _parse_query_transform(
    raw_text: str,
    *,
    fallback_query: str,
    max_queries: int,
) -> QueryTransformResult:
    payload = parse_json_object(raw_text)
    if payload is None:
        return QueryTransformResult(standalone_query=fallback_query, search_queries=(fallback_query,))

    standalone = str(payload.get("standalone_query") or fallback_query).strip() or fallback_query
    raw_queries = payload.get("search_queries")
    queries: list[str] = []
    seen: set[str] = set()
    if isinstance(raw_queries, list):
        for item in raw_queries:
            q = str(item or "").strip()
            low = q.lower()
            if q and low not in seen:
                seen.add(low)
                queries.append(q)
                if len(queries) >= max_queries:
                    break

    if not queries:
        queries = [standalone]

    return QueryTransformResult(standalone_query=standalone, search_queries=tuple(queries))


def build_rag_nodes(
    *,
    llm: ChatRunnable,
    retrieval_engine: RetrievalEngine,
    agent_config: RagAgentConfig,
) -> dict[str, Callable[..., Awaitable[dict[str, Any]]]]:
    """Build pure async LangGraph node callables for `rag_agent`."""

    async def transform_query(state: RagState, config: RunnableConfig) -> dict[str, Any]:
        query = str(state.get("query") or "")
        attempt = state.get("attempt", 0) + 1
        prior = state.get("search_queries") or []
        tail = state.get("conversation_tail") or []
        tail_block = "Recent context:\n" + "\n".join(tail) + "\n\n" if tail else ""

        sys_prompt = build_query_transform_prompt(
            attempt=attempt,
            max_queries=agent_config.max_queries,
            prior_queries=prior,
        )
        response = await llm.ainvoke(
            [
                SystemMessage(content=sys_prompt),
                HumanMessage(content=f"{tail_block}Current request: {query}"),
            ],
            config=with_nostream(config),
        )
        parsed = _parse_query_transform(
            extract_text(response.content),
            fallback_query=query,
            max_queries=agent_config.max_queries,
        )
        prior_usage = state.get("usage") or RunUsage()
        return {
            "attempt": attempt,
            "standalone_query": parsed.standalone_query,
            "search_queries": list(parsed.search_queries),
            "usage": prior_usage.merge(RunUsage.from_message(response)),
        }

    async def retrieve_passages(state: RagState, config: RunnableConfig) -> dict[str, Any]:
        del config
        base_query = str(state.get("query") or "")
        standalone = str(state.get("standalone_query") or base_query)
        space_id = str(state.get("space_id") or "")
        search_queries = state.get("search_queries") or [standalone]
        batch = await retrieval_engine.retrieve(
            RetrievalQuery(
                space_id=space_id,
                primary_query=standalone,
                expanded_queries=tuple(search_queries),
                source_types=tuple(state.get("source_types") or ()),
                source_ids=tuple(state.get("source_ids") or ()),
            )
        )
        return {
            "passages": list(batch.passages),
            "ranked_desc": list(batch.ranked_desc),
            "total_candidates": batch.total_candidates,
        }

    async def evaluate_crag(state: RagState) -> dict[str, Any]:
        ranked_desc = state.get("ranked_desc") or []
        if len(ranked_desc) >= agent_config.min_passages:
            return {"abstain": False}
        if state.get("attempt", 1) < agent_config.max_attempts:
            return {"abstain": False}
        return {"abstain": True}

    async def generate_answer(state: RagState, config: RunnableConfig) -> dict[str, Any]:
        ranked_desc = state.get("ranked_desc") or []
        passages_repacked = state.get("passages") or []
        if state.get("abstain", False) or not ranked_desc:
            return {
                "final_answer": ABSTENTION_RESPONSE,
                "citations": [],
                "abstain": True,
            }

        query = str(state.get("standalone_query") or state.get("query") or "")
        user_prompt = format_evidence_block(
            query=query,
            passages_repacked=passages_repacked,
            ranked_desc=ranked_desc,
        )
        response = await llm.ainvoke(
            [
                SystemMessage(content=build_grounded_answer_prompt()),
                HumanMessage(content=user_prompt),
            ],
            config=config,
        )
        answer_text = strip_reasoning(extract_text(response.content))
        citations = select_indexed_citations(
            answer_text,
            [passage.to_citation() for passage in ranked_desc],
        )
        prior_usage = state.get("usage") or RunUsage()
        return {
            "final_answer": answer_text,
            "citations": citations,
            "abstain": False,
            "usage": prior_usage.merge(RunUsage.from_message(response)),
        }

    return {
        "transform_query": transform_query,
        "retrieve_passages": retrieve_passages,
        "evaluate_crag": evaluate_crag,
        "generate_answer": generate_answer,
    }
