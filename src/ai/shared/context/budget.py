"""Dynamic Token Budgeting & Knapsack Allocation Policy.

Applies the Policy Pattern to enforce hard context window budgets across
heterogeneous domains and ensure zero token overflow.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ai.shared.context.contracts import EntityDomain, GraphNode


class TokenBudgetPolicy:
    """Manages token quota distribution and safe-slicing across research domains."""

    def __init__(
        self,
        *,
        max_context_tokens: int = 4000,
        primacy_ratio: float = 0.20,
        middle_ratio: float = 0.50,
        recency_ratio: float = 0.30,
        chars_per_token: float = 4.0,
    ) -> None:
        if max_context_tokens < 100:
            raise ValueError("max_context_tokens must be >= 100")
        total_ratio = primacy_ratio + middle_ratio + recency_ratio
        if not (0.99 <= total_ratio <= 1.01):
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

        self._max_tokens = max_context_tokens
        self._primacy_ratio = primacy_ratio
        self._middle_ratio = middle_ratio
        self._recency_ratio = recency_ratio
        self._chars_per_token = max(1.0, chars_per_token)

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def primacy_budget(self) -> int:
        return int(self._max_tokens * self._primacy_ratio)

    @property
    def middle_budget(self) -> int:
        return int(self._max_tokens * self._middle_ratio)

    @property
    def recency_budget(self) -> int:
        return int(self._max_tokens * self._recency_ratio)

    def estimate_tokens(self, text: str) -> int:
        """Conservative token count estimation based on character length."""
        if not text:
            return 0
        return max(1, int(len(text) / self._chars_per_token))

    def fit_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget while keeping full lines intact."""
        clean = text.strip()
        if not clean or max_tokens <= 0:
            return ""

        current_tokens = self.estimate_tokens(clean)
        if current_tokens <= max_tokens:
            return clean

        max_chars = int(max_tokens * self._chars_per_token)
        truncated = clean[:max_chars]
        last_newline = truncated.rfind("\n")
        if last_newline > int(max_chars * 0.6):
            truncated = truncated[:last_newline]
        return truncated + "\n...[truncated to budget]"

    def allocate_nodes(
        self,
        nodes: Sequence[GraphNode],
        domain_weights: Mapping[EntityDomain, float],
        max_budget: int | None = None,
    ) -> list[GraphNode]:
        """Greedy knapsack allocation of graph nodes by domain quota.

        Ensures balanced representation across Governance, Literature, Experiments,
        and Pages according to the active projection strategy.
        """
        budget = max_budget if max_budget is not None else self.middle_budget
        if budget <= 0 or not nodes:
            return []

        # Calculate per-domain token quotas
        domain_quotas: dict[EntityDomain, int] = {}
        for domain, weight in domain_weights.items():
            domain_quotas[domain] = max(1, int(budget * max(0.0, weight)))

        # Sort nodes by priority / density descending (preserve topological/BFS order on ties)
        scored_nodes: list[tuple[float, int, GraphNode, int]] = []
        for idx, node in enumerate(nodes):
            desc = f"{node.label}: {node.attributes}"
            node_tokens = self.estimate_tokens(desc)
            raw_prio = node.attributes.get("priority", node.attributes.get("score", 1.0))
            try:
                prio = float(raw_prio)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                prio = 1.0
            scored_nodes.append((prio, -idx, node, node_tokens))

        scored_nodes.sort(key=lambda item: (item[0], item[1]), reverse=True)

        allocated: list[GraphNode] = []
        allocated_ids: set[str] = set()
        overflow_candidates: list[tuple[GraphNode, int]] = []
        domain_usage: dict[EntityDomain, int] = {d: 0 for d in EntityDomain}
        total_used = 0

        # Pass 1: Respect per-domain quotas
        for _, _, node, node_tokens in scored_nodes:
            if node.id in allocated_ids:
                continue
            quota = domain_quotas.get(node.domain, int(budget * 0.25))
            current_usage = domain_usage[node.domain]

            if current_usage + node_tokens <= quota and total_used + node_tokens <= budget:
                allocated.append(node)
                allocated_ids.add(node.id)
                domain_usage[node.domain] += node_tokens
                total_used += node_tokens
            elif total_used + node_tokens <= budget:
                overflow_candidates.append((node, node_tokens))

        # Pass 2: Work-conserving slack redistribution (fill remaining budget from overflow)
        for node, node_tokens in overflow_candidates:
            if node.id not in allocated_ids and total_used + node_tokens <= budget:
                allocated.append(node)
                allocated_ids.add(node.id)
                total_used += node_tokens

        return allocated
