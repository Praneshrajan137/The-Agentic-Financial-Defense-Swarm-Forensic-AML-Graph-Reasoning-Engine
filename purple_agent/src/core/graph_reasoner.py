"""
Graph Reasoner -- Forensic Graph Traversal Engine
PRD Reference: Task B1
Package Version: v8.0

Treats the financial network as a MultiDiGraph (multiple directed edges per node pair).
BFS for 1-hop neighbor discovery (structuring detection).
DFS for chain traversal (layering detection).

CRITICAL: ALL amounts stored as Decimal. NEVER float.
CRITICAL: nx.MultiDiGraph, NEVER nx.DiGraph.
CRITICAL: assert is disabled by python -O. Use raise TypeError for type guards.
CRITICAL: DFS is ITERATIVE (stack-based), NEVER recursive (Rule 20).

v8.0: Three critical corrections over v7.0:
  - Empty transactions -> ValueError (Rule 19, Step 2.5) [P4v8-03]
  - Currency None normalization -> "USD" default [P4v8-04]
  - Super-node check uses TOTAL degree, not directional [P4v8-06]

v7.0: Full Architecture S4 Step 2.5 validation suite at ingestion boundary.
v6.1: Iterative DFS, NaN guard, path explosion protection.
"""
import logging
from decimal import Decimal

import networkx as nx

from src.config import MAX_DFS_DEPTH, MAX_PATHS_PER_SEARCH, MAX_NODE_DEGREE

logger = logging.getLogger(__name__)


class GraphReasoner:
    """
    The forensic graph reasoning engine.

    Uses MultiDiGraph to preserve all parallel transactions between entities.
    Provides BFS (1-hop) for structuring fan-in/fan-out detection and
    DFS (chain traversal) for layering detection with decay analysis.

    v8.0: Total-degree super-node protection (Rule 21). Empty TX rejection (Rule 19).
    v7.0: Full Architecture S4 Step 2.5 ingestion validation suite.
    v6.1: Iterative DFS with explicit stack (Rule 20). NaN guard (Rule 2).
    """

    def __init__(self) -> None:
        """Initialize with empty MultiDiGraph."""
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

    def _check_super_node(self, node_id: str) -> bool:
        """
        Check if a node exceeds the super-node degree limit.

        v8.0 [P4v8-06]: Uses TOTAL degree (in_degree + out_degree) per
        Architecture Rule 21: "skip nodes with degree > MAX_NODE_DEGREE."
        v7.0 used directional degree -- a node with in=300, out=300,
        total=600 > 500 would have passed both directional checks.

        Args:
            node_id: Node to check.

        Returns:
            True if the node exceeds MAX_NODE_DEGREE (should be skipped).
        """
        total_degree = self.graph.degree(node_id)
        if total_degree > MAX_NODE_DEGREE:
            logger.warning(
                f"Super-node detected: {node_id} has total degree {total_degree} "
                f"(limit: {MAX_NODE_DEGREE}). Skipping to prevent OOM. "
                f"(in={self.graph.in_degree(node_id)}, "
                f"out={self.graph.out_degree(node_id)})"
            )
            return True
        return False

    def load_from_dict(self, graph_data: dict) -> None:
        """
        Load a graph from dict (deserialized Protobuf or test fixture).

        Implements the COMPLETE Architecture S4 Step 2.5 validation suite:
          1. Empty transactions -> ValueError (Rule 19) [v8.0 P4v8-03]
          2. Decimal type check with TypeError on violation (Rule 1)
          3. .is_finite() check with ValueError on NaN/Infinity (Rule 2)
          4. Currency None normalization -> "USD" default [v8.0 P4v8-04]
          5. Empty-string currency -> ValueError (Step 2.5) [v7.0 P4-05]
          6. Duplicate tx_id: keep FIRST, skip subsequent, log warning (Rule 27)
          7. Self-loop: LOG info, do NOT reject (intra-bank transfers)
          8. Timestamp <= 0: WARN, do NOT reject (defensive pattern)
          9. Post-loop edge count check -> ValueError if 0 edges loaded

        Args:
            graph_data: Dict with "transactions" (list) and "nodes" (dict).

        Raises:
            TypeError: If any transaction amount is not a Decimal.
            ValueError: If transactions list is empty, any amount is NaN/Infinity,
                        currency is empty string, or zero edges loaded after processing.
        """
        if self.graph.number_of_nodes() > 0:
            logger.warning(
                "load_from_dict called on non-empty graph "
                f"({self.graph.number_of_nodes()} nodes, "
                f"{self.graph.number_of_edges()} edges). Clearing all data."
            )
        self.graph.clear()

        transactions = graph_data.get("transactions", [])
        if not transactions:
            raise ValueError(
                "GraphFragment contains zero transactions. "
                "An empty graph produces false negatives (clearing criminals). "
                "This is treated as a fetch failure per Rule 19."
            )

        for node_id, attrs in sorted(graph_data.get("nodes", {}).items()):
            self.graph.add_node(node_id, **attrs)

        seen_tx_ids: set[str] = set()

        for tx in transactions:
            tx_id = tx["id"]

            if tx_id in seen_tx_ids:
                logger.warning(
                    f"Duplicate tx_id '{tx_id}': keeping first occurrence, "
                    f"skipping duplicate (source={tx['source_node']}, "
                    f"target={tx['target_node']}, amount={tx['amount']}). "
                    f"Rule 27: NEVER silently overwrite."
                )
                continue
            seen_tx_ids.add(tx_id)

            if not isinstance(tx["amount"], Decimal):
                raise TypeError(
                    f"Transaction {tx_id}: amount must be Decimal, "
                    f"got {type(tx['amount']).__name__} with value {tx['amount']}"
                )

            if not tx["amount"].is_finite():
                raise ValueError(
                    f"Transaction {tx_id}: amount must be finite, "
                    f"got {tx['amount']} (NaN and Infinity are forbidden)"
                )

            # v7.0 [P4-05]: Reject explicit empty-string currency BEFORE
            # normalization. An entity that explicitly sets currency="" is a
            # data quality error that must be surfaced, not silently defaulted.
            raw_currency = tx.get("currency")
            if raw_currency is not None and raw_currency == "":
                raise ValueError(
                    f"Transaction {tx_id}: currency must be non-empty. "
                    f"Provide a valid ISO currency code (e.g., 'USD', 'INR')."
                )
            # v8.0 [P4v8-04]: Normalize None/missing-key → "USD".
            currency = raw_currency or "USD"

            timestamp = tx.get("timestamp", 0)
            if timestamp <= 0:
                logger.warning(
                    f"Transaction {tx_id}: timestamp={timestamp} is <= 0. "
                    f"This may indicate missing or corrupt temporal data. "
                    f"Proceeding with defensive default (not rejecting)."
                )

            source = tx["source_node"]
            target = tx["target_node"]
            if source == target:
                logger.info(
                    f"Transaction {tx_id}: self-loop detected "
                    f"(source_node == target_node == '{source}'). "
                    f"Retaining for analysis (may be intra-bank transfer or layering hop)."
                )

            self.graph.add_edge(
                source,
                target,
                key=tx_id,
                tx_id=tx_id,
                amount=tx["amount"],
                currency=currency,
                timestamp=timestamp,
                tx_type=tx.get("type", "WIRE"),
                branch_code=tx.get("branch_code", ""),
            )

        if self.graph.number_of_edges() == 0:
            raise ValueError(
                f"GraphFragment contained {len(transactions)} transaction(s) but "
                f"zero edges were loaded (all may have been duplicates). "
                f"An empty graph produces false negatives per Rule 19."
            )

    def get_1hop_incoming(self, node_id: str) -> list[dict]:
        """
        BFS 1-hop: Get all incoming transactions to a node.

        Args:
            node_id: Target node to find incoming transactions for.

        Returns:
            List of transaction dicts sorted by tx_id for determinism.
        """
        if node_id not in self.graph:
            return []

        if self._check_super_node(node_id):
            return []

        incoming: list[dict] = []
        for predecessor in self.graph.predecessors(node_id):
            for edge_key, edge_data in self.graph[predecessor][node_id].items():
                incoming.append({
                    "tx_id": edge_data["tx_id"],
                    "source_node": predecessor,
                    "target_node": node_id,
                    "amount": edge_data["amount"],
                    "currency": edge_data["currency"],
                    "timestamp": edge_data["timestamp"],
                    "tx_type": edge_data["tx_type"],
                    "branch_code": edge_data.get("branch_code", ""),
                })
        return sorted(incoming, key=lambda x: x["tx_id"])

    def get_1hop_outgoing(self, node_id: str) -> list[dict]:
        """
        BFS 1-hop: Get all outgoing transactions from a node.

        Args:
            node_id: Source node to find outgoing transactions for.

        Returns:
            List of transaction dicts sorted by tx_id for determinism.
        """
        if node_id not in self.graph:
            return []

        if self._check_super_node(node_id):
            return []

        outgoing: list[dict] = []
        for successor in self.graph.successors(node_id):
            for edge_key, edge_data in self.graph[node_id][successor].items():
                outgoing.append({
                    "tx_id": edge_data["tx_id"],
                    "source_node": node_id,
                    "target_node": successor,
                    "amount": edge_data["amount"],
                    "currency": edge_data["currency"],
                    "timestamp": edge_data["timestamp"],
                    "tx_type": edge_data["tx_type"],
                    "branch_code": edge_data.get("branch_code", ""),
                })
        return sorted(outgoing, key=lambda x: x["tx_id"])

    def dfs_trace_chains(
        self, start_node: str, max_depth: int | None = None
    ) -> list[list[dict]]:
        """
        DFS: Trace all outflow chains from start_node.

        Returns list of chains (each chain = list of transaction dicts).

        v6.1 [ALN-03]: ITERATIVE implementation using explicit stack.
        Rule 20: "DFS must be iterative (stack-based), never recursive"

        v7.0 [P4-15]: Traversal order is deterministic: alphabetical by tx_id.
        get_1hop_outgoing returns sorted results. reversed() push order ensures
        the alphabetically-first tx_id is popped (explored) first from the LIFO stack.

        Args:
            start_node: Node ID to begin chain traversal from.
            max_depth: Maximum DFS depth. Defaults to MAX_DFS_DEPTH from config.

        Returns:
            List of chains. Each chain is a list of transaction dicts.
        """
        if max_depth is None:
            max_depth = MAX_DFS_DEPTH

        if start_node not in self.graph:
            return []

        all_chains: list[list[dict]] = []
        initial_visited = frozenset({start_node})
        stack: list[tuple[str, list[dict], frozenset, int]] = [
            (start_node, [], initial_visited, 0)
        ]

        while stack:
            if len(all_chains) >= MAX_PATHS_PER_SEARCH:
                break

            current_node, current_chain, visited, depth = stack.pop()

            if depth >= max_depth:
                if current_chain:
                    all_chains.append(current_chain)
                continue

            outgoing = self.get_1hop_outgoing(current_node)

            if not outgoing:
                if current_chain:
                    all_chains.append(current_chain)
                continue

            unvisited_txs = [
                tx for tx in outgoing
                if tx["target_node"] not in visited
            ]

            if not unvisited_txs:
                if current_chain:
                    all_chains.append(current_chain)
                continue

            for tx in reversed(unvisited_txs):
                if len(all_chains) >= MAX_PATHS_PER_SEARCH:
                    if current_chain:
                        all_chains.append(current_chain)
                    break

                next_node = tx["target_node"]
                new_chain = current_chain + [tx]
                new_visited = visited | {next_node}
                stack.append((next_node, new_chain, new_visited, depth + 1))

        if len(all_chains) >= MAX_PATHS_PER_SEARCH:
            logger.warning(
                f"DFS from {start_node}: path limit reached ({MAX_PATHS_PER_SEARCH}). "
                "Results are truncated. Possible adversarial graph structure."
            )

        return all_chains

    def get_node_attributes(self, node_id: str) -> dict | None:
        """Get attributes for a node, or None if not found."""
        if node_id not in self.graph:
            return None
        return dict(self.graph.nodes[node_id])

    def get_subgraph(self, node_ids: list[str]) -> nx.MultiDiGraph:
        """Extract a read-only COPY of the subgraph for the specified nodes."""
        valid_nodes = [n for n in node_ids if n in self.graph]
        return self.graph.subgraph(valid_nodes).copy()

    def get_all_node_ids(self) -> list[str]:
        """Return all node IDs sorted alphabetically (Rule 15)."""
        return sorted(self.graph.nodes())

    def has_node(self, node_id: str) -> bool:
        """Check whether a node ID exists in the graph."""
        return node_id in self.graph

    def get_edge_count(self) -> int:
        """Return total number of edges (transactions) in the graph."""
        return self.graph.number_of_edges()

    def get_node_count(self) -> int:
        """Return total number of nodes (entities) in the graph."""
        return self.graph.number_of_nodes()
