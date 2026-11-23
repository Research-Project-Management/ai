"""Chain of Responsibility resolver mapping user names/slugs into exact Backend UUIDs."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from ai.agents.project.gateway.contracts import (
    ProjectLabel,
    ProjectMember,
    ProjectSummary,
    WorkflowState,
)
from ai.agents.project.resolution.contracts import (
    CandidateItem,
    ResolutionResult,
    ResolvedEntity,
)

_STATE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "backlog": ("backlog", "icebox", "triage"),
    "unstarted": ("todo", "to do", "unstarted", "open", "ready"),
    "started": ("in progress", "in-progress", "doing", "active", "started", "wip"),
    "completed": ("done", "completed", "complete", "finished", "closed"),
    "cancelled": ("cancelled", "canceled", "wontfix", "dropped"),
}


class EntityMatchStrategy(Protocol):
    """Chain link returning matching `CandidateItem` entries for a raw query."""

    name: str

    def match(self, query: str, candidates: Sequence[CandidateItem]) -> list[CandidateItem]: ...


class ExactIdStrategy:
    """First-pass exact UUID / canonical identifier match (case-insensitive)."""

    name = "exact_id"

    def match(self, query: str, candidates: Sequence[CandidateItem]) -> list[CandidateItem]:
        q = query.strip().lower()
        return [c for c in candidates if c.id.strip().lower() == q]


class ExactNameOrAliasStrategy:
    """Exact case-insensitive match on candidate display name or aliases."""

    name = "exact_name"

    def match(self, query: str, candidates: Sequence[CandidateItem]) -> list[CandidateItem]:
        q = query.strip().lower()
        hits: list[CandidateItem] = []
        for cand in candidates:
            pool = (cand.name, *cand.aliases)
            if any(token.strip().lower() == q for token in pool if token.strip()):
                hits.append(cand)
        return hits


class SubstringPrefixStrategy:
    """Fallback prefix, then case-insensitive substring match over names & aliases."""

    name = "fuzzy_match"

    def match(self, query: str, candidates: Sequence[CandidateItem]) -> list[CandidateItem]:
        q = query.strip().lower()
        if len(q) < 2:
            return []

        prefix_hits: list[CandidateItem] = []
        sub_hits: list[CandidateItem] = []
        for cand in candidates:
            tokens = [t.strip().lower() for t in (cand.name, *cand.aliases) if t.strip()]
            if any(tok.startswith(q) for tok in tokens):
                prefix_hits.append(cand)
            elif any(q in tok for tok in tokens):
                sub_hits.append(cand)

        return prefix_hits if prefix_hits else sub_hits


_WORD_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ỹ0-9]{2,}")


def _token_set(text: str) -> set[str]:
    return {w.lower() for w in _WORD_TOKEN_RE.findall(text)}


class TokenSubsetStrategy:
    """Final fallback matching when all query tokens appear in a candidate's name or aliases."""

    name = "fuzzy_match"

    def match(self, query: str, candidates: Sequence[CandidateItem]) -> list[CandidateItem]:
        q_tokens = _token_set(query)
        if not q_tokens:
            return []
        hits: list[CandidateItem] = []
        for cand in candidates:
            cand_tokens = _token_set(f"{cand.name} {' '.join(cand.aliases)}")
            if q_tokens <= cand_tokens:
                hits.append(cand)
        return hits


DEFAULT_ENTITY_STRATEGIES: tuple[EntityMatchStrategy, ...] = (
    ExactIdStrategy(),
    ExactNameOrAliasStrategy(),
    SubstringPrefixStrategy(),
    TokenSubsetStrategy(),
)


def resolve_entity(
    query: str,
    candidates: Sequence[CandidateItem],
    *,
    entity_type: str,
    strategies: Sequence[EntityMatchStrategy] = DEFAULT_ENTITY_STRATEGIES,
) -> ResolutionResult:
    """Resolve user query to one `CandidateItem` using the strategy chain."""
    clean = query.strip()
    if not clean:
        return ResolutionResult(ok=False, error=f"Empty {entity_type} reference.")
    if not candidates:
        return ResolutionResult(ok=False, error=f"No {entity_type} options exist in this project.")

    for strategy in strategies:
        hits = strategy.match(clean, candidates)
        if len(hits) == 1:
            chosen = hits[0]
            return ResolutionResult(
                ok=True,
                match=ResolvedEntity(id=chosen.id, label=chosen.name, strategy=strategy.name),
            )
        if len(hits) > 1:
            names = ", ".join(f"'{h.name}'" for h in hits[:5])
            return ResolutionResult(
                ok=False,
                ambiguous=tuple(hits[:5]),
                error=(
                    f"Ambiguous {entity_type} '{clean}' matches multiple entries: {names}. "
                    "Specify the exact name or ID."
                ),
            )

    available = ", ".join(f"'{c.name}'" for c in candidates[:10])
    return ResolutionResult(
        ok=False,
        error=f"{entity_type.capitalize()} '{clean}' not found. Available: {available}.",
    )


def resolve_member(query: str, members: Sequence[ProjectMember]) -> ResolutionResult:
    """Resolve a team member by `user_id`, full name, or email prefix."""
    candidates = [
        CandidateItem(
            id=m.user_id,
            name=m.name,
            aliases=tuple(
                tok for tok in (m.email, m.email.split("@", 1)[0] if "@" in m.email else "") if tok
            ),
        )
        for m in members
    ]
    return resolve_entity(query, candidates, entity_type="member")


def resolve_workflow_state(query: str, states: Sequence[WorkflowState]) -> ResolutionResult:
    """Resolve a Kanban column by `state.id`, column name, `group`, or common synonym."""
    candidates: list[CandidateItem] = []
    for st in states:
        synonyms = _STATE_SYNONYMS.get(st.group.lower(), ())
        aliases = tuple(dict.fromkeys((st.group, *synonyms)))
        candidates.append(CandidateItem(id=st.id, name=st.name, aliases=aliases))
    return resolve_entity(query, candidates, entity_type="workflow column")


def resolve_label(query: str, labels: Sequence[ProjectLabel]) -> ResolutionResult:
    """Resolve a project label by `label.id` or name."""
    candidates = [CandidateItem(id=lb.id, name=lb.name) for lb in labels]
    return resolve_entity(query, candidates, entity_type="label")


def resolve_project(query: str, projects: Sequence[ProjectSummary]) -> ResolutionResult:
    """Resolve a project by `id`, `identifier` (e.g. FLUX), or name."""
    candidates = [
        CandidateItem(
            id=p.id,
            name=p.name,
            aliases=(p.identifier,) if p.identifier else (),
        )
        for p in projects
    ]
    return resolve_entity(query, candidates, entity_type="project")
