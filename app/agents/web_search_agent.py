"""Web Search Agent — searches the web (DuckDuckGo + OpenAlex) and synthesises results."""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage

from app.agents.base import AgentState, BaseAgent
from app.agents.registry import register_agent
from app.config import get_llm

logger = logging.getLogger(__name__)

DEFAULT_ACADEMIC_SITES = [
    "arxiv.org",
    "aclanthology.org",
    "openreview.net",
    "ieeexplore.ieee.org",
    "dl.acm.org",
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


def _build_query(user_query: str, sites: list[str]) -> str:
    """Return the bare user query for DuckDuckGo (no site: filters — they break multi-site OR in DDG)."""
    return user_query


def _translate_to_english(text: str) -> str:
    """Translate a query to English via LLM so OpenAlex (English-indexed) returns better results."""
    try:
        llm = get_llm(temperature=0)
        response = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Translate the following academic search query to English. "
                        "Return ONLY the translated query, nothing else. "
                        "If it is already in English, return it unchanged."
                    ),
                },
                {"role": "user", "content": text},
            ]
        )
        translated = response.content.strip()
        logger.info("Translated OpenAlex query: %r → %r", text, translated)
        return translated
    except Exception as exc:
        logger.warning("Query translation failed, using original: %s", exc)
        return text


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
    """Merge DuckDuckGo + OpenAlex results, deduplicated by URL."""
    _sites = sites or DEFAULT_ACADEMIC_SITES
    # Extract bare user query (strip site: filters) for OpenAlex
    bare_query = query.split(")", 1)[-1].strip() if ")" in query else query
    # OpenAlex is English-indexed — translate before searching
    english_query = _translate_to_english(bare_query)

    ddg = _search_ddg(query, max_results=max_results)
    # Filter DDG results to only include URLs that belong to the configured sites
    ddg = [r for r in ddg if any(site in r.get("href", "") for site in _sites)]
    oalex = _search_openalex(english_query, sites=_sites, max_results=_OPENALEX_MAX)

    # Deduplicate by URL
    seen: set[str] = set()
    merged: list[dict] = []
    for r in ddg + oalex:
        url = r.get("href", "")
        if url and url in seen:
            continue
        if url:
            seen.add(url)
        merged.append(r)

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

        query = _build_query(user_query, sites)
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
        metadata = state.get("metadata", {})
        sites: list[str] = metadata.get("web_search_sites") or DEFAULT_ACADEMIC_SITES

        results, query = self.fetch_results(state)

        formatted = _format_results(results)

        # Synthesize with LLM
        llm = get_llm()
        synthesis_messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\n"
                    f"Search results:\n{formatted}"
                ),
            },
        ]
        response = await llm.ainvoke(synthesis_messages)
        content = response.content if isinstance(response.content, str) else str(response.content)

        sources = [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")[:200]}
            for r in results
        ]

        state["agent_output"] = {
            "agent_name": self.name,
            "content": content,
            "sources": sources,
            "metadata": {"query": query, "sites": sites},
        }
        state["messages"] = [AIMessage(content=content)]
        return state
