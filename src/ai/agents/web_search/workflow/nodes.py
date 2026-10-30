"""Async LangGraph node callables for `web_search_agent`."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ai.agents.web_search.fetch.contracts import TERMINAL_SECURITY_BLOCKS, FetchRequest
from ai.agents.web_search.fetch.extractor import extract_academic_identifiers
from ai.agents.web_search.fetch.ladder import UrlFetchLadder
from ai.agents.web_search.ranking.contracts import BlockedSourceReport
from ai.agents.web_search.ranking.deduplicator import (
    canonicalize_url,
    compute_canonical_id,
    deduplicate_hits,
)
from ai.agents.web_search.ranking.triage import (
    curate_evidence_batch,
    score_search_hit,
    select_urls_for_deep_fetch,
)
from ai.agents.web_search.search.contracts import SearchHit, SearchQuery
from ai.agents.web_search.search.registry import SearchProviderRegistry
from ai.agents.web_search.workflow.contracts import (
    WEB_ABSTENTION_RESPONSE,
    SearchMode,
    SearchPlan,
    WebSearchAgentConfig,
)
from ai.agents.web_search.workflow.prompts import (
    build_search_planner_prompt,
    build_web_synthesis_prompt,
    format_web_evidence_block,
)
from ai.agents.web_search.workflow.state import WebSearchState
from ai.orchestrator.citations import select_indexed_citations
from ai.shared.contracts.run_control import RunUsage
from ai.shared.llm.factory import ChatRunnable
from ai.shared.llm.nostream import with_nostream
from ai.shared.llm.output import extract_text, parse_json_object, strip_reasoning

_VALID_MODES: frozenset[str] = frozenset({"direct_url", "scholar", "web", "hybrid"})


def _parse_search_plan(
    raw_text: str,
    *,
    fallback_query: str,
    max_sub_queries: int,
    has_seed_urls: bool,
    mode_override: SearchMode | None,
) -> SearchPlan:
    default_mode: SearchMode = (
        mode_override
        if mode_override is not None
        else ("direct_url" if has_seed_urls else "hybrid")
    )
    payload = parse_json_object(raw_text)
    if payload is None:
        return SearchPlan(
            standalone_query=fallback_query,
            mode=default_mode,
            sub_queries=() if default_mode == "direct_url" else (fallback_query,),
        )

    standalone = str(payload.get("standalone_query") or fallback_query).strip() or fallback_query
    raw_mode = str(payload.get("mode") or default_mode).strip().lower()
    if mode_override is not None:
        resolved_mode: SearchMode = mode_override
    elif raw_mode in _VALID_MODES:
        resolved_mode = cast(SearchMode, raw_mode)
    else:
        resolved_mode = default_mode

    if resolved_mode == "direct_url" and not has_seed_urls:
        resolved_mode = "hybrid"

    raw_queries = payload.get("sub_queries")
    queries: list[str] = []
    seen: set[str] = set()
    if isinstance(raw_queries, list):
        for item in raw_queries:
            q = str(item or "").strip()
            low = q.lower()
            if q and low not in seen:
                seen.add(low)
                queries.append(q)
                if len(queries) >= max_sub_queries:
                    break

    if resolved_mode != "direct_url" and not queries:
        queries = [standalone]

    return SearchPlan(
        standalone_query=standalone,
        mode=resolved_mode,
        sub_queries=tuple(queries),
    )


def _format_abstention_with_blocks(blocked_sources: Sequence[BlockedSourceReport]) -> str:
    if not blocked_sources:
        return WEB_ABSTENTION_RESPONSE
    details = "\n".join(f"- {b.url} ({b.reason})" for b in blocked_sources)
    return (
        f"{WEB_ABSTENTION_RESPONSE}\n\n"
        f"Các nguồn bị chặn hoặc không thể truy cập trực tiếp:\n{details}"
    )


def build_web_search_nodes(
    *,
    llm: ChatRunnable,
    search_registry: SearchProviderRegistry,
    fetch_ladder: UrlFetchLadder,
    agent_config: WebSearchAgentConfig,
) -> dict[str, Callable[..., Awaitable[dict[str, Any]]]]:
    """Assemble the 4 async nodes for the `web_search_agent` LangGraph subgraph."""

    async def plan_search(state: WebSearchState, config: RunnableConfig) -> dict[str, Any]:
        query = str(state.get("query") or "").strip()
        round_index = int(state.get("round_index", 0)) + 1
        seed_urls = state.get("seed_urls") or []
        prior_queries = state.get("sub_queries") or []
        tail = state.get("conversation_tail") or []
        mode_override = state.get("search_mode_override")

        # Fast-path: caller pinned direct_url reading with explicit URLs/client sources
        if round_index == 1 and mode_override == "direct_url" and seed_urls:
            return {
                "round_index": round_index,
                "standalone_query": query,
                "planned_mode": "direct_url",
                "sub_queries": [],
            }

        tail_block = "Recent context:\n" + "\n".join(tail) + "\n\n" if tail else ""
        sys_prompt = build_search_planner_prompt(
            round_index=round_index,
            max_sub_queries=agent_config.max_sub_queries,
            has_seed_urls=bool(seed_urls),
            prior_queries=prior_queries,
            mode_override=mode_override,
        )
        response = await llm.ainvoke(
            [
                SystemMessage(content=sys_prompt),
                HumanMessage(content=f"{tail_block}Current request: {query}"),
            ],
            config=with_nostream(config),
        )
        parsed = _parse_search_plan(
            extract_text(response.content),
            fallback_query=query,
            max_sub_queries=agent_config.max_sub_queries,
            has_seed_urls=bool(seed_urls),
            mode_override=mode_override if round_index == 1 else None,
        )
        prior_usage = state.get("usage") or RunUsage()
        return {
            "round_index": round_index,
            "standalone_query": parsed.standalone_query,
            "planned_mode": parsed.mode,
            "sub_queries": list(parsed.sub_queries),
            "usage": prior_usage.merge(RunUsage.from_message(response)),
        }

    async def execute_search_and_fetch(
        state: WebSearchState, config: RunnableConfig
    ) -> dict[str, Any]:
        del config
        base_query = str(state.get("query") or "")
        standalone = str(state.get("standalone_query") or base_query).strip()
        mode: SearchMode = state.get("planned_mode") or "hybrid"
        sub_queries = list(state.get("sub_queries") or ([standalone] if standalone else []))
        seed_urls = tuple(state.get("seed_urls") or ())
        allowed_domains = tuple(state.get("allowed_domains") or ())
        blocked_domains = tuple(state.get("blocked_domains") or ())
        client_sources = tuple(state.get("client_sources") or ())

        round_hits: tuple[SearchHit, ...] = ()
        if mode != "direct_url" and sub_queries:
            category = "scholar" if mode == "scholar" else "web" if mode == "web" else "hybrid"
            queries = [
                SearchQuery(
                    query=q,
                    category=category,
                    max_results=agent_config.max_results_per_provider,
                    allowed_domains=allowed_domains,
                    blocked_domains=blocked_domains,
                )
                for q in sub_queries
            ]
            round_hits = await search_registry.search_many(
                queries,
                timeout_seconds=agent_config.search_timeout_seconds,
            )

        all_hits = deduplicate_hits([*(state.get("raw_hits") or []), *round_hits])
        scored = [(hit, score_search_hit(standalone, hit)) for hit in all_hits]
        to_fetch_hits = select_urls_for_deep_fetch(
            scored,
            max_deep_fetch=agent_config.max_deep_fetch_urls,
            seed_urls=seed_urls,
        )

        # Build deduplicated fetch requests across explicit seed URLs, caller-supplied client URLs,
        # and snippet-insufficient search candidates.
        fetch_requests: list[FetchRequest] = []
        queued_keys: set[str] = set()

        def _enqueue(url: str, *, identifiers_hint: Any = None) -> None:
            canon_url = canonicalize_url(url) or url.strip()
            derived = extract_academic_identifiers(url)
            merged_ids = derived.merge(identifiers_hint) if identifiers_hint is not None else derived
            cid = compute_canonical_id(url=url, doi=merged_ids.doi, arxiv_id=merged_ids.arxiv_id)
            if cid in queued_keys or canon_url in queued_keys:
                return
            queued_keys.add(cid)
            queued_keys.add(canon_url)
            fetch_requests.append(
                FetchRequest(
                    url=url,
                    identifiers=merged_ids,
                    client_sources=client_sources,
                    allowed_domains=allowed_domains,
                    blocked_domains=blocked_domains,
                )
            )

        for url in seed_urls:
            _enqueue(url)
        for src in client_sources:
            _enqueue(src.url)
        for hit in to_fetch_hits:
            ids_from_hit = extract_academic_identifiers(hit.url)
            if hit.doi or hit.arxiv_id or hit.pmid:
                ids_from_hit = ids_from_hit.merge(
                    ids_from_hit.model_copy(
                        update={
                            "doi": ids_from_hit.doi or hit.doi,
                            "arxiv_id": ids_from_hit.arxiv_id or hit.arxiv_id,
                            "pmid": ids_from_hit.pmid or hit.pmid,
                        }
                    )
                )
            _enqueue(hit.url, identifiers_hint=ids_from_hit)

        fetch_results = await fetch_ladder.fetch_many(fetch_requests)
        batch = curate_evidence_batch(
            query=standalone,
            hits=all_hits,
            fetch_results=fetch_results,
            seed_urls=seed_urls,
            top_k=agent_config.top_k_evidence,
            min_score=agent_config.min_relevance_score,
        )
        return {
            "raw_hits": all_hits,
            "repacked_evidence": list(batch.repacked),
            "ranked_desc": list(batch.ranked_desc),
            "blocked_sources": list(batch.blocked_sources),
        }

    async def evaluate_coverage(state: WebSearchState) -> dict[str, Any]:
        ranked = state.get("ranked_desc") or []
        if len(ranked) >= agent_config.min_evidence_count:
            return {"abstain": False}

        blocked = state.get("blocked_sources") or []
        if blocked and all(item.reason in TERMINAL_SECURITY_BLOCKS for item in blocked):
            return {"abstain": True}

        if int(state.get("round_index", 1)) < agent_config.max_rounds:
            return {"abstain": False}
        return {"abstain": True}

    async def synthesize_answer(state: WebSearchState, config: RunnableConfig) -> dict[str, Any]:
        ranked_desc = state.get("ranked_desc") or []
        repacked = state.get("repacked_evidence") or []
        blocked = state.get("blocked_sources") or []
        if state.get("abstain", False) or not ranked_desc:
            return {
                "final_answer": _format_abstention_with_blocks(blocked),
                "citations": [],
                "abstain": True,
            }

        raw_query = str(state.get("query") or "").strip()
        standalone = str(state.get("standalone_query") or "").strip()
        if raw_query and standalone and raw_query.lower() != standalone.lower():
            query = f"{raw_query}\n(Resolved search context: {standalone})"
        else:
            query = raw_query or standalone
        user_prompt = format_web_evidence_block(
            query=query,
            repacked=repacked,
            ranked_desc=ranked_desc,
            blocked_sources=blocked,
        )
        response = await llm.ainvoke(
            [
                SystemMessage(content=build_web_synthesis_prompt()),
                HumanMessage(content=user_prompt),
            ],
            config=config,
        )
        answer_text = strip_reasoning(extract_text(response.content))
        citations = select_indexed_citations(
            answer_text,
            [item.to_citation() for item in ranked_desc],
        )
        prior_usage = state.get("usage") or RunUsage()
        return {
            "final_answer": answer_text,
            "citations": citations,
            "abstain": False,
            "usage": prior_usage.merge(RunUsage.from_message(response)),
        }

    return {
        "plan_search": plan_search,
        "execute_search_and_fetch": execute_search_and_fetch,
        "evaluate_coverage": evaluate_coverage,
        "synthesize_answer": synthesize_answer,
    }
