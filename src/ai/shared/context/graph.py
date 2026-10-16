"""In-memory Directed Acyclic Graph (DAG) for Scientific Provenance & Lineage.

Applies the Graph / Flyweight Pattern for fast k-hop BFS traversal and bidirectional
lineage tracing across Governance, Literature, Experiments, and Pages.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence

from ai.shared.context.contracts import EntityDomain, GraphEdge, GraphNode, RelationType


class ProvenanceGraph:
    """Represents the interconnected multi-relational scientific research graph."""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        # Adjacency list: src_id -> list[GraphEdge]
        self._out_edges: dict[str, list[GraphEdge]] = defaultdict(list)
        # Reverse adjacency list: dst_id -> list[GraphEdge]
        self._in_edges: dict[str, list[GraphEdge]] = defaultdict(list)

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self._out_edges.values())

    @property
    def nodes(self) -> list[GraphNode]:
        return list(self._nodes.values())

    def add_node(self, node: GraphNode) -> None:
        """Register or update an entity node."""
        self._nodes[node.id] = node

    def add_edge(
        self,
        src_id: str,
        dst_id: str,
        relation: RelationType,
        metadata: dict[str, object] | None = None,
    ) -> GraphEdge:
        """Create a directed provenance relation edge between two nodes."""
        edge = GraphEdge(
            src_id=src_id,
            dst_id=dst_id,
            relation=relation,
            metadata=dict(metadata or {}),
        )
        for idx, existing in enumerate(self._out_edges[src_id]):
            if existing.dst_id == dst_id and existing.relation == relation:
                self._out_edges[src_id][idx] = edge
                for in_idx, in_existing in enumerate(self._in_edges[dst_id]):
                    if in_existing.src_id == src_id and in_existing.relation == relation:
                        self._in_edges[dst_id][in_idx] = edge
                        break
                return edge
        self._out_edges[src_id].append(edge)
        self._in_edges[dst_id].append(edge)
        return edge

    def get_node(self, node_id: str) -> GraphNode | None:
        """Look up a node by its unique identifier."""
        return self._nodes.get(node_id)

    def get_nodes_by_domain(self, domain: EntityDomain) -> list[GraphNode]:
        """Filter nodes belonging to a specific research domain."""
        return [n for n in self._nodes.values() if n.domain == domain]

    def get_out_edges(self, node_id: str) -> list[GraphEdge]:
        """Retrieve all outgoing edges originating from this node."""
        return list(self._out_edges.get(node_id, ()))

    def get_in_edges(self, node_id: str) -> list[GraphEdge]:
        """Retrieve all incoming edges terminating at this node."""
        return list(self._in_edges.get(node_id, ()))

    def has_edge(
        self,
        src_id: str,
        dst_id: str,
        relation: RelationType | None = None,
    ) -> bool:
        """Check if an edge exists between two nodes, optionally matching relation type."""
        for edge in self._out_edges.get(src_id, ()):
            if edge.dst_id == dst_id and (relation is None or edge.relation == relation):
                return True
        return False

    def get_k_hop_subgraph(
        self,
        root_id: str,
        k: int = 1,
        domain_filter: Sequence[EntityDomain] | None = None,
    ) -> list[GraphNode]:
        """Breadth-first search exploring bidirectional semantic neighbors up to radius k.

        Grounded in GraphRAG community detection: extracts local subgraphs to bound
        token consumption without losing cross-domain connections.
        """
        if root_id not in self._nodes or k < 1:
            return []

        allowed_domains = set(domain_filter) if domain_filter is not None else None
        visited: set[str] = {root_id}
        queue: deque[tuple[str, int]] = deque([(root_id, 0)])
        result: list[GraphNode] = []

        while queue:
            curr_id, depth = queue.popleft()
            if curr_id != root_id and curr_id in self._nodes:
                node = self._nodes[curr_id]
                if allowed_domains is None or node.domain in allowed_domains:
                    result.append(node)

            if depth < k:
                # Explore both forward and backward connections
                neighbors: set[str] = set()
                for edge in self._out_edges.get(curr_id, ()):
                    neighbors.add(edge.dst_id)
                for edge in self._in_edges.get(curr_id, ()):
                    neighbors.add(edge.src_id)

                for neighbor_id in neighbors:
                    if neighbor_id not in visited and neighbor_id in self._nodes:
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, depth + 1))

        return result

    def get_upstream_lineage(self, target_id: str) -> list[GraphNode]:
        """Trace the backward chain of custody (e.g. Page <- Run <- Task <- Paper).

        Follows incoming PROVES, TRACKS, INFORMS, and DERIVED_FROM edges.
        """
        if target_id not in self._nodes:
            return []

        lineage: list[GraphNode] = []
        visited: set[str] = {target_id}
        queue: deque[str] = deque([target_id])

        upstream_relations = {
            RelationType.PROVES,
            RelationType.TRACKS,
            RelationType.INFORMS,
            RelationType.DERIVED_FROM,
        }

        while queue:
            curr_id = queue.popleft()
            for edge in self._in_edges.get(curr_id, ()):
                if edge.relation in upstream_relations and edge.src_id not in visited:
                    visited.add(edge.src_id)
                    src_node = self._nodes.get(edge.src_id)
                    if src_node is not None:
                        lineage.append(src_node)
                        queue.append(edge.src_id)

        return lineage
