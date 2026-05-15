"""Web Search Agent — searches the web (DuckDuckGo + OpenAlex) and synthesises results."""

from __future__ import annotations

import logging

from app.agents.base import AgentState, BaseAgent
from app.agents.registry import register_agent

logger = logging.getLogger(__name__)

DEFAULT_ACADEMIC_SITES = [
    # Primary academic sources
    "arxiv.org",
    "ieeexplore.ieee.org",
    "dl.acm.org",
    "pubmed.ncbi.nlm.nih.gov",
    "semanticscholar.org",
    "scholar.google.com",
    # Publishers
    "springer.com",
    "nature.com",
    "sciencedirect.com",
    "researchgate.net",
    # Conference proceedings & preprints
    "aclanthology.org",
    "openreview.net",
    "zenodo.org",
    "ceur-ws.org",
    "proceedings.mlr.press",
    "proceedings.neurips.cc",
    "openaccess.thecvf.com",
    "jmlr.org",
    "biorxiv.org",
    "medrxiv.org",
]

_MAX_RESULTS = 8
_OPENALEX_MAX = 5


# Top-priority sites for DDG site:-targeted searches
_DDG_PRIORITY_SITES = ["arxiv.org", "scholar.google.com", "semanticscholar.org"]


def _build_site_queries(user_query: str, sites: list[str]) -> list[str]:
    """Build multiple DDG queries: one per priority site (with site: operator) + one broad query.

    DDG supports `site:domain` for single-domain filtering which is much more
    reliable than post-hoc URL filtering. We issue a few targeted queries in
    parallel for the sites most likely to have good results.
    """
    queries = []
    # Pick up to 3 priority sites that are in the user's allowed list
    priority = [s for s in _DDG_PRIORITY_SITES if s in sites][:3]
    for site in priority:
        queries.append(f"{user_query} site:{site}")
    # Always include a broad query (will be post-filtered)
    queries.append(user_query)
    return queries


from app.agents.utils import _translate_to_english


def _search_ddg(query: str, max_results: int = _MAX_RESULTS) -> list[dict]:
    """Run a DuckDuckGo text search and return normalised result dicts."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        logger.info("DuckDuckGo returned %d results for query: %s", len(raw), query)
        return [
            {"title": r.get("title", ""), "href": r.get("href", ""), "body": r.get("body", ""), "source": "ddg"}
            for r in raw
        ]
    except Exception as exc:
        logger.warning("DuckDuckGo search failed: %s", exc)
        return []


def _search_openalex(query: str, sites: list[str], max_results: int = _OPENALEX_MAX) -> list[dict]:
    """
    Search OpenAlex works API (https://openalex.org/).
    Only includes results whose landing page URL matches one of the allowed sites.
    """
    try:
        import httpx
        params = {
            "search": query,
            "per_page": max_results * 3,  # fetch more to compensate for filtering
            "select": "title,doi,primary_location,abstract_inverted_index,publication_year,authorships",
            "mailto": "flux-ai@research",
        }
        resp = httpx.get("https://api.openalex.org/works", params=params, timeout=10)
        resp.raise_for_status()
        works = resp.json().get("results", [])

        def _matches_sites(url: str) -> bool:
            if not url or not sites:
                return True
            return any(site in url for site in sites)

        results = []
        for w in works:
            # Prefer the actual landing page (e.g. arxiv.org/abs/...) over bare DOI
            primary = w.get("primary_location") or {}
            landing = primary.get("landing_page_url") or ""
            doi = w.get("doi") or ""
            doi_url = doi if doi.startswith("http") else (f"https://doi.org/{doi}" if doi else "")
            url = landing or doi_url

            # Skip if the URL doesn't belong to any allowed site
            if not _matches_sites(url):
                continue

            # Reconstruct abstract from inverted index
            abstract = ""
            inv = w.get("abstract_inverted_index") or {}
            if inv:
                words: list[tuple[int, str]] = []
                for word, positions in inv.items():
                    for pos in positions:
                        words.append((pos, word))
                abstract = " ".join(wd for _, wd in sorted(words))[:400]

            year = w.get("publication_year") or ""
            authors_list = [
                a["author"]["display_name"]
                for a in (w.get("authorships") or [])[:3]
                if a.get("author")
            ]
            authors = ", ".join(authors_list)
            body = abstract + (f" — {authors}" if authors else "") + (f" ({year})" if year else "")

            doi = w.get("doi") or ""
            doi_url = doi if doi.startswith("http") else (f"https://doi.org/{doi}" if doi else "")

            results.append({
                "title": w.get("title") or "Untitled",
                "href": url,
                "body": body.strip(),
                "source": "openalex",
                # Rich metadata kept as separate fields for the sources panel
                "authors": authors or None,
                "year": int(year) if year else None,
                "doi": doi_url or None,
            })
            if len(results) >= max_results:
                break

        logger.info("OpenAlex returned %d works (after site filter)", len(results))
        return results
    except Exception as exc:
        logger.warning("OpenAlex search failed: %s", exc)
        return []


def _search(query: str, sites: list[str] | None = None, max_results: int = _MAX_RESULTS) -> list[dict]:
    """Merge DuckDuckGo + OpenAlex results, deduplicated by URL.

    Strategy:
    1. Issue targeted DDG queries with site: operator for priority sites
    2. Issue a broad DDG query and post-filter by allowed sites
    3. If both produce 0 results, return the unfiltered broad DDG results as fallback
    4. Always run OpenAlex in parallel for academic enrichment
    """
    _sites = sites or DEFAULT_ACADEMIC_SITES
    user_has_sites = sites is not None and len(sites) > 0

    # OpenAlex is English-indexed — translate before searching
    english_query = _translate_to_english(query)

    # ── DDG searches: targeted + broad ──────────────────────────
    ddg_queries = _build_site_queries(query, _sites)
    all_ddg: list[dict] = []
    broad_ddg: list[dict] = []  # unfiltered broad results (last query)

    for i, q in enumerate(ddg_queries):
        results = _search_ddg(q, max_results=max_results)
        is_broad = (i == len(ddg_queries) - 1)  # last query is always the broad one

        if is_broad:
            broad_ddg = results[:]
            # Post-filter broad results by allowed sites (only if user specified sites)
            if user_has_sites:
                results = [r for r in results if any(site in r.get("href", "") for site in _sites)]
        # targeted queries already have site: so all results are relevant
        all_ddg.extend(results)

    # ── OpenAlex ──────────────────────────────────────────────────
    oalex = _search_openalex(english_query, sites=_sites, max_results=_OPENALEX_MAX)

    # ── Merge + deduplicate ───────────────────────────────────────
    seen: set[str] = set()
    merged: list[dict] = []
    for r in all_ddg + oalex:
        url = r.get("href", "")
        if url and url in seen:
            continue
        if url:
            seen.add(url)
        merged.append(r)

    # ── Fallback: if everything was filtered away, return unfiltered broad results
    if not merged and broad_ddg:
        logger.warning("All DDG results filtered out — returning %d unfiltered broad results", len(broad_ddg))
        merged = broad_ddg

    logger.info("Web search merged: %d results (ddg_targeted=%d, broad=%d, openalex=%d)",
                len(merged), len(all_ddg), len(broad_ddg), len(oalex))
    return merged


def _format_results(results: list[dict]) -> str:
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        href = r.get("href", "")
        body = r.get("body", "")[:300]
        source_tag = "[OpenAlex]" if r.get("source") == "openalex" else "[Web]"
        lines.append(f"{source_tag} [{i}] {title}\nURL: {href}\nSnippet: {body}\n")
    return "\n".join(lines)


SYNTHESIS_SYSTEM_PROMPT = """\
You are a research assistant. The user asked a question and the following \
search results were retrieved from web sources and OpenAlex. \
Write a well-structured answer that:
1. Summarises the KEY FINDINGS and main points from the snippets — do NOT just list titles and links.
2. Groups related information into short paragraphs or bullet points.
3. Cites sources inline using markdown links like [Paper Title](url) so they render as clickable badges.
4. Ends with a "## Sources" section listing all references as markdown links.
If the results are not sufficient to fully answer, say so and share what you do know.
"""


@register_agent
class WebSearchAgent(BaseAgent):
    """
    Web search agent that queries academic/scientific sources and
    synthesises the results with the LLM.
    """

    name = "web_search"
    description = (
        "Search the web for up-to-date information, research papers, "
        "scientific articles, and academic publications. Use this when the "
        "user asks about recent research, wants citations, needs information "
        "beyond the uploaded documents, or explicitly asks to search online."
    )

    def fetch_results(self, state: AgentState) -> tuple[list[dict], str]:
        """Run the search and return (results, raw_query). Used by the streaming endpoint."""
        metadata = state.get("metadata", {})
        sites: list[str] = metadata.get("web_search_sites") or DEFAULT_ACADEMIC_SITES

        messages = state.get("messages", [])
        user_query = ""
        for msg in reversed(messages):
            role = getattr(msg, "type", None) or getattr(msg, "role", "")
            if role in ("human", "user"):
                user_query = msg.content if hasattr(msg, "content") else str(msg)
                break
        if not user_query and messages:
            user_query = str(messages[-1].content)

        query = user_query
        logger.info("WebSearch query: %s", query)
        results = _search(query, sites=sites)
        return results, query

    def fetch_results_with_query(
        self, search_query: str, sites: list[str]
    ) -> tuple[list[dict], str]:
        """Run search with a pre-optimised query (built by chat.py's _rewrite_query).

        Preferred over fetch_results for the streaming pipeline — the caller
        already has a concise, search-ready string so we skip the message extraction step.
        """
        _sites = sites or DEFAULT_ACADEMIC_SITES
        logger.info("WebSearch (pre-built query): %s | sites: %s", search_query, _sites)
        results = _search(search_query, sites=_sites)
        return results, search_query

    async def run(self, state: AgentState) -> AgentState:
        """Run web search and synthesise results into agent_output.

        This bridges the BaseAgent interface so WebSearchAgent can be registered
        in the registry and discovered by the Supervisor.  The streaming pipeline
        (graph.py web_research_node) calls fetch_results / fetch_results_with_query
        directly for real-time token streaming; this method handles the non-streaming
        (sync) path used by /chat/sync and any future direct invocations.
        """
        from app.config import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage

        results, query = self.fetch_results(state)
        formatted = _format_results(results)

        llm = get_llm()
        synthesis_messages = [
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {query}\n\nSearch Results:\n{formatted}"),
        ]
        try:
            response = await llm.ainvoke(synthesis_messages)
            content = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            logger.error("[WebSearchAgent] LLM synthesis failed: %s", exc)
            content = f"Web search found {len(results)} results but synthesis failed: {exc}\n\n{formatted}"

        sources = [
            {"source": r.get("title", r.get("href", "")), "url": r.get("href", ""), "snippet": r.get("body", "")[:200]}
            for r in results
        ]

        return {
            **state,
            "agent_output": {
                "agent_name": self.name,
                "content": content,
                "sources": sources,
                "metadata": {"query": query, "result_count": len(results)},
            },
            "intent": "web_search",
        }

