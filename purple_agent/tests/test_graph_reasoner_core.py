"""
Tests for Graph Reasoner Core -- Purple Agent v8.0
PRD Reference: Task B1
TDD: Tests written FIRST, implementation SECOND.

Tests cover:
  - Graph loading (node/edge counts, Decimal amounts)
  - BFS 1-hop (incoming, outgoing, empty predecessors)
  - DFS chain tracing (paths, max_depth, path explosion)
  - Super-node protection (total degree, directional-below-total-above)
  - Node attributes (found, not found)
  - Scenario loading (structuring, layering)
  - MultiDiGraph parallel edges preserved
  - Determinism (sorted node IDs)
  - Warnings (non-empty graph reload)
  - Validation (float TypeError, NaN/Infinity ValueError)
  - Iterative DFS correctness
  - v7.0 Step 2.5 (duplicate tx_id, self-loop, empty currency, zero timestamp)
  - v8.0 (None currency, empty transactions, all-duplicates post-loop check)
"""
import logging
from decimal import Decimal
from unittest.mock import patch

import pytest

from src.core.graph_reasoner import GraphReasoner
from src.config import MAX_NODE_DEGREE, MAX_PATHS_PER_SEARCH


# ===================================================================
# Graph Loading Tests
# ===================================================================

class TestGraphLoading:

    def test_load_graph_node_count(self, structuring_scenario: dict) -> None:
        gr = GraphReasoner()
        gr.load_from_dict(structuring_scenario)
        assert gr.get_node_count() == len(structuring_scenario["nodes"])

    def test_load_graph_edge_count(self, structuring_scenario: dict) -> None:
        gr = GraphReasoner()
        gr.load_from_dict(structuring_scenario)
        expected = len(structuring_scenario["transactions"])
        assert gr.get_edge_count() == expected

    def test_amounts_are_decimal(self, structuring_scenario: dict) -> None:
        gr = GraphReasoner()
        gr.load_from_dict(structuring_scenario)
        incoming = gr.get_1hop_incoming("mule_1")
        for tx in incoming:
            if not isinstance(tx["amount"], Decimal):
                raise TypeError(
                    f"Amount {tx['amount']} is {type(tx['amount']).__name__}, "
                    f"expected Decimal"
                )

    def test_load_structuring_scenario(self, structuring_scenario: dict) -> None:
        gr = GraphReasoner()
        gr.load_from_dict(structuring_scenario)
        assert gr.has_node("mule_1")
        for src in structuring_scenario["criminal_sources"]:
            assert gr.has_node(src)

    def test_load_layering_scenario(self, layering_scenario: dict) -> None:
        gr = GraphReasoner()
        gr.load_from_dict(layering_scenario)
        for node_id in layering_scenario["chain_nodes"]:
            assert gr.has_node(node_id)
        assert gr.get_edge_count() == layering_scenario["hop_count"]


# ===================================================================
# BFS 1-Hop Tests
# ===================================================================

class TestBFS1Hop:

    def test_1hop_incoming(self, structuring_scenario: dict) -> None:
        gr = GraphReasoner()
        gr.load_from_dict(structuring_scenario)
        incoming = gr.get_1hop_incoming("mule_1")
        sources = sorted(set(tx["source_node"] for tx in incoming))
        expected_sources = sorted(
            structuring_scenario["criminal_sources"]
            + structuring_scenario["legitimate_nodes"]
        )
        assert sources == expected_sources

    def test_1hop_outgoing(self, layering_scenario: dict) -> None:
        gr = GraphReasoner()
        gr.load_from_dict(layering_scenario)
        outgoing = gr.get_1hop_outgoing("layer_origin")
        assert len(outgoing) == 1
        assert outgoing[0]["target_node"] == "layer_1"
        assert outgoing[0]["amount"] == Decimal("100000")

    def test_1hop_incoming_empty(self, layering_scenario: dict) -> None:
        gr = GraphReasoner()
        gr.load_from_dict(layering_scenario)
        incoming = gr.get_1hop_incoming("layer_origin")
        assert incoming == []


# ===================================================================
# DFS Chain Tracing Tests
# ===================================================================

class TestDFSChainTracing:

    def test_dfs_trace_chain(self, layering_scenario: dict) -> None:
        gr = GraphReasoner()
        gr.load_from_dict(layering_scenario)
        chains = gr.dfs_trace_chains("layer_origin")
        assert len(chains) >= 1
        longest = max(chains, key=len)
        assert len(longest) == 4

    def test_dfs_respects_max_depth(self, layering_scenario: dict) -> None:
        gr = GraphReasoner()
        gr.load_from_dict(layering_scenario)
        chains = gr.dfs_trace_chains("layer_origin", max_depth=2)
        for chain in chains:
            assert len(chain) <= 2

    def test_dfs_path_explosion_protection(self) -> None:
        """Clique graph: N nodes fully connected. Chains capped at MAX_PATHS_PER_SEARCH."""
        gr = GraphReasoner()
        n = 12
        nodes = {f"clq_{i}": {"name": f"Clique {i}"} for i in range(n)}
        txs = []
        tx_counter = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    tx_counter += 1
                    txs.append({
                        "id": f"TX-CLQ-{tx_counter:04d}",
                        "source_node": f"clq_{i}",
                        "target_node": f"clq_{j}",
                        "amount": Decimal("1000"),
                        "currency": "USD",
                        "timestamp": 1700000000 + tx_counter,
                        "type": "WIRE",
                    })
        gr.load_from_dict({"transactions": txs, "nodes": nodes})
        chains = gr.dfs_trace_chains("clq_0")
        assert len(chains) <= MAX_PATHS_PER_SEARCH

    def test_iterative_dfs_matches_expected_chains(self, layering_scenario: dict) -> None:
        """Verify iterative DFS produces correct chain through the layering scenario."""
        gr = GraphReasoner()
        gr.load_from_dict(layering_scenario)
        chains = gr.dfs_trace_chains("layer_origin")
        assert len(chains) == 1
        chain = chains[0]
        assert len(chain) == 4
        expected_targets = ["layer_1", "layer_2", "layer_3", "layer_4"]
        actual_targets = [tx["target_node"] for tx in chain]
        assert actual_targets == expected_targets


# ===================================================================
# Super-Node Tests
# ===================================================================

class TestSuperNode:

    def test_super_node_skipped(self, super_node_scenario: dict) -> None:
        gr = GraphReasoner()
        gr.load_from_dict(super_node_scenario)
        incoming = gr.get_1hop_incoming("hub_node")
        assert incoming == []

    def test_super_node_directional_below_total_above(self) -> None:
        """v8.0 [P4v8-06]: in=300 out=300 total=600 > MAX_NODE_DEGREE=500 -> skipped."""
        gr = GraphReasoner()
        nodes = {"center": {"name": "Center"}}
        txs = []
        for i in range(1, 301):
            src_id = f"in_{i:04d}"
            nodes[src_id] = {"name": f"In {i}"}
            txs.append({
                "id": f"TX-IN-{i:04d}",
                "source_node": src_id,
                "target_node": "center",
                "amount": Decimal("1000"),
                "currency": "USD",
                "timestamp": 1700000000 + i,
                "type": "WIRE",
            })
        for i in range(1, 301):
            tgt_id = f"out_{i:04d}"
            nodes[tgt_id] = {"name": f"Out {i}"}
            txs.append({
                "id": f"TX-OUT-{i:04d}",
                "source_node": "center",
                "target_node": tgt_id,
                "amount": Decimal("1000"),
                "currency": "USD",
                "timestamp": 1700000000 + 300 + i,
                "type": "WIRE",
            })
        gr.load_from_dict({"transactions": txs, "nodes": nodes})
        assert gr.graph.in_degree("center") == 300
        assert gr.graph.out_degree("center") == 300
        assert gr.graph.degree("center") == 600
        assert gr.get_1hop_incoming("center") == []
        assert gr.get_1hop_outgoing("center") == []


# ===================================================================
# Node Attribute Tests
# ===================================================================

class TestNodeAttributes:

    def test_get_node_attributes(self, structuring_scenario: dict) -> None:
        gr = GraphReasoner()
        gr.load_from_dict(structuring_scenario)
        attrs = gr.get_node_attributes("mule_1")
        assert attrs is not None
        assert attrs["name"] == "Shell Corp LLC"

    def test_get_node_attributes_nonexistent(self, structuring_scenario: dict) -> None:
        gr = GraphReasoner()
        gr.load_from_dict(structuring_scenario)
        assert gr.get_node_attributes("nonexistent_node") is None


# ===================================================================
# MultiDiGraph Tests
# ===================================================================

class TestMultiDiGraph:

    def test_multi_edge_preserved(self, multi_edge_scenario: dict) -> None:
        """CRITICAL: 3 edges A->B, MultiDiGraph preserves all 3."""
        gr = GraphReasoner()
        gr.load_from_dict(multi_edge_scenario)
        assert gr.get_edge_count() == 3
        outgoing = gr.get_1hop_outgoing("A")
        assert len(outgoing) == 3
        amounts = sorted(tx["amount"] for tx in outgoing)
        assert amounts == [Decimal("9100"), Decimal("9200"), Decimal("9300")]


# ===================================================================
# Determinism Tests
# ===================================================================

class TestDeterminism:

    def test_get_all_node_ids_sorted(self, structuring_scenario: dict) -> None:
        gr = GraphReasoner()
        gr.load_from_dict(structuring_scenario)
        node_ids = gr.get_all_node_ids()
        assert node_ids == sorted(node_ids)


# ===================================================================
# Warning Tests
# ===================================================================

class TestWarnings:

    def test_load_from_dict_warns_on_nonempty_graph(
        self, structuring_scenario: dict, caplog: pytest.LogCaptureFixture,
    ) -> None:
        gr = GraphReasoner()
        gr.load_from_dict(structuring_scenario)
        with caplog.at_level(logging.WARNING):
            gr.load_from_dict(structuring_scenario)
        assert any("non-empty graph" in r.message for r in caplog.records)


# ===================================================================
# Validation Tests (Type / Value errors)
# ===================================================================

class TestValidation:

    def test_float_amount_raises_type_error(self) -> None:
        gr = GraphReasoner()
        data = {
            "transactions": [{
                "id": "TX-FLOAT-1",
                "source_node": "A",
                "target_node": "B",
                "amount": 9500.0,
                "currency": "USD",
                "timestamp": 1700000000,
                "type": "WIRE",
            }],
            "nodes": {"A": {"name": "A"}, "B": {"name": "B"}},
        }
        with pytest.raises(TypeError, match="amount must be Decimal"):
            gr.load_from_dict(data)

    def test_nan_amount_raises_value_error(self) -> None:
        gr = GraphReasoner()
        data = {
            "transactions": [{
                "id": "TX-NAN-1",
                "source_node": "A",
                "target_node": "B",
                "amount": Decimal("NaN"),
                "currency": "USD",
                "timestamp": 1700000000,
                "type": "WIRE",
            }],
            "nodes": {"A": {"name": "A"}, "B": {"name": "B"}},
        }
        with pytest.raises(ValueError, match="must be finite"):
            gr.load_from_dict(data)

    def test_infinity_amount_raises_value_error(self) -> None:
        gr = GraphReasoner()
        data = {
            "transactions": [{
                "id": "TX-INF-1",
                "source_node": "A",
                "target_node": "B",
                "amount": Decimal("Infinity"),
                "currency": "USD",
                "timestamp": 1700000000,
                "type": "WIRE",
            }],
            "nodes": {"A": {"name": "A"}, "B": {"name": "B"}},
        }
        with pytest.raises(ValueError, match="must be finite"):
            gr.load_from_dict(data)


# ===================================================================
# v7.0 Step 2.5 Validation Tests
# ===================================================================

class TestStep25Validation:

    def test_duplicate_tx_id_keeps_first(self, duplicate_txid_scenario: dict) -> None:
        gr = GraphReasoner()
        gr.load_from_dict(duplicate_txid_scenario)
        assert gr.get_edge_count() == duplicate_txid_scenario["expected_tx_count_after_dedup"]
        outgoing_from_a = gr.get_1hop_outgoing("A")
        ab_txs = [tx for tx in outgoing_from_a if tx["target_node"] == "B"]
        assert len(ab_txs) == 1
        assert ab_txs[0]["amount"] == Decimal("9100")

    def test_duplicate_tx_id_logs_warning(
        self, duplicate_txid_scenario: dict, caplog: pytest.LogCaptureFixture,
    ) -> None:
        gr = GraphReasoner()
        with caplog.at_level(logging.WARNING):
            gr.load_from_dict(duplicate_txid_scenario)
        dup_warnings = [r for r in caplog.records if "Duplicate tx_id" in r.message]
        assert len(dup_warnings) == duplicate_txid_scenario["expected_warnings"]

    def test_self_loop_logged_not_rejected(
        self, self_loop_scenario: dict, caplog: pytest.LogCaptureFixture,
    ) -> None:
        gr = GraphReasoner()
        with caplog.at_level(logging.INFO):
            gr.load_from_dict(self_loop_scenario)
        assert gr.get_edge_count() == self_loop_scenario["expected_total_tx"]
        loop_logs = [r for r in caplog.records if "self-loop" in r.message]
        assert len(loop_logs) == self_loop_scenario["expected_self_loops"]

    def test_empty_currency_raises_value_error(self) -> None:
        gr = GraphReasoner()
        data = {
            "transactions": [{
                "id": "TX-EMPTY-CUR",
                "source_node": "A",
                "target_node": "B",
                "amount": Decimal("9100"),
                "currency": "",
                "timestamp": 1700000000,
                "type": "WIRE",
            }],
            "nodes": {"A": {"name": "A"}, "B": {"name": "B"}},
        }
        with pytest.raises(ValueError, match="currency must be non-empty"):
            gr.load_from_dict(data)

    def test_zero_timestamp_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        gr = GraphReasoner()
        data = {
            "transactions": [{
                "id": "TX-ZERO-TS",
                "source_node": "A",
                "target_node": "B",
                "amount": Decimal("5000"),
                "currency": "USD",
                "timestamp": 0,
                "type": "WIRE",
            }],
            "nodes": {"A": {"name": "A"}, "B": {"name": "B"}},
        }
        with caplog.at_level(logging.WARNING):
            gr.load_from_dict(data)
        assert gr.get_edge_count() == 1
        ts_warnings = [r for r in caplog.records if "timestamp" in r.message and "<= 0" in r.message]
        assert len(ts_warnings) == 1


# ===================================================================
# v8.0 Tests
# ===================================================================

class TestV8Enhancements:

    def test_none_currency_defaults_to_usd(self) -> None:
        """v8.0 [P4v8-04]: None currency -> 'USD' default."""
        gr = GraphReasoner()
        data = {
            "transactions": [{
                "id": "TX-NONE-CUR",
                "source_node": "A",
                "target_node": "B",
                "amount": Decimal("9100"),
                "currency": None,
                "timestamp": 1700000000,
                "type": "WIRE",
            }],
            "nodes": {"A": {"name": "A"}, "B": {"name": "B"}},
        }
        gr.load_from_dict(data)
        outgoing = gr.get_1hop_outgoing("A")
        assert len(outgoing) == 1
        assert outgoing[0]["currency"] == "USD"

    def test_empty_transactions_raises_value_error(self, empty_graph_scenario: dict) -> None:
        """v8.0 [P4v8-03]: Empty transactions list -> ValueError."""
        gr = GraphReasoner()
        with pytest.raises(ValueError, match="zero transactions"):
            gr.load_from_dict(empty_graph_scenario)

    def test_all_duplicates_raises_value_error(self) -> None:
        """v8.0 [P4v8-03]: Post-loop check catches zero edges after processing.

        The dedup logic always keeps the first occurrence, so we monkeypatch
        add_edge to be a no-op to exercise the defensive post-loop check.
        """
        gr = GraphReasoner()
        data = {
            "transactions": [
                {
                    "id": "TX-ALL-DUP-1",
                    "source_node": "A",
                    "target_node": "B",
                    "amount": Decimal("5000"),
                    "currency": "USD",
                    "timestamp": 1700000000,
                    "type": "WIRE",
                },
            ],
            "nodes": {"A": {"name": "A"}, "B": {"name": "B"}},
        }
        with patch.object(gr.graph, "add_edge"):
            with pytest.raises(ValueError, match="zero edges were loaded"):
                gr.load_from_dict(data)
