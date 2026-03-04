"""
Tests for Layering Detection -- Purple Agent v8.0
PRD Reference: Task B3
TDD: Tests written FIRST, implementation SECOND.

Tests cover:
  - Chain detection (3% decay detected, hop_count >= MIN_CHAIN_LENGTH)
  - Decay rate validation (within [0.02, 0.05] tolerance band)
  - Amounts are Decimal throughout
  - Confidence scoring (nonzero for detected chains)
  - Negative cases (no decay, excessive decay, short chain, slow chain, increasing amounts)
  - Multiple chains detection
  - v8.0: Mixed currency chain rejection, currency recorded in result
"""
from decimal import Decimal

import pytest

from src.core.graph_reasoner import GraphReasoner
from src.core.heuristics.layering import detect_layering, LayeringChain, LayeringResult
from src.config import MIN_CHAIN_LENGTH


# ===================================================================
# Helper
# ===================================================================

def _load(scenario: dict) -> GraphReasoner:
    gr = GraphReasoner()
    gr.load_from_dict(scenario)
    return gr


# ===================================================================
# Detection Tests
# ===================================================================

class TestLayeringDetection:

    def test_detects_layering_chain(self, layering_scenario: dict) -> None:
        gr = _load(layering_scenario)
        result = detect_layering(gr, "layer_origin")
        assert result.detected is True
        assert result.total_chains_found >= 1

    def test_hop_count_meets_minimum(self, layering_scenario: dict) -> None:
        gr = _load(layering_scenario)
        result = detect_layering(gr, "layer_origin")
        for chain in result.chains:
            assert chain.hop_count >= MIN_CHAIN_LENGTH

    def test_decay_rate_in_range(self, layering_scenario: dict) -> None:
        """Decay rates within [0.02, 0.05] +/- tolerance."""
        gr = _load(layering_scenario)
        result = detect_layering(gr, "layer_origin")
        for chain in result.chains:
            for rate in chain.decay_rates:
                assert Decimal("0.015") <= rate <= Decimal("0.055"), (
                    f"Decay rate {rate} outside tolerance band"
                )

    def test_chain_nodes_correct(self, layering_scenario: dict) -> None:
        gr = _load(layering_scenario)
        result = detect_layering(gr, "layer_origin")
        chain = result.chains[0]
        expected_nodes = layering_scenario["chain_nodes"]
        assert chain.chain_nodes == expected_nodes

    def test_amounts_are_decimal(self, layering_scenario: dict) -> None:
        gr = _load(layering_scenario)
        result = detect_layering(gr, "layer_origin")
        chain = result.chains[0]
        if not isinstance(chain.start_amount, Decimal):
            raise TypeError(f"start_amount is {type(chain.start_amount)}, expected Decimal")
        if not isinstance(chain.end_amount, Decimal):
            raise TypeError(f"end_amount is {type(chain.end_amount)}, expected Decimal")
        for rate in chain.decay_rates:
            if not isinstance(rate, Decimal):
                raise TypeError(f"decay rate is {type(rate)}, expected Decimal")

    def test_confidence_score_nonzero(self, layering_scenario: dict) -> None:
        gr = _load(layering_scenario)
        result = detect_layering(gr, "layer_origin")
        assert result.confidence > Decimal("0")


# ===================================================================
# Negative Cases
# ===================================================================

class TestNegativeCases:

    def test_no_decay_not_detected(self) -> None:
        """Equal amounts along chain = NOT layering (0% decay outside range)."""
        gr = GraphReasoner()
        nodes = {f"nd_{i}": {"name": f"Node {i}"} for i in range(5)}
        txs = []
        for i in range(4):
            txs.append({
                "id": f"TX-NODEC-{i+1}",
                "source_node": f"nd_{i}",
                "target_node": f"nd_{i+1}",
                "amount": Decimal("100000"),
                "currency": "USD",
                "timestamp": 1700000000 + (i * 7200),
                "type": "WIRE",
            })
        gr.load_from_dict({"transactions": txs, "nodes": nodes})
        result = detect_layering(gr, "nd_0")
        assert result.detected is False

    def test_excessive_decay_not_detected(self) -> None:
        """>5.5% decay: outside range + tolerance."""
        gr = GraphReasoner()
        nodes = {f"ex_{i}": {"name": f"Node {i}"} for i in range(5)}
        amounts = [Decimal("100000"), Decimal("92000"), Decimal("84640"), Decimal("77868.80")]
        txs = []
        for i in range(4):
            txs.append({
                "id": f"TX-EXDEC-{i+1}",
                "source_node": f"ex_{i}",
                "target_node": f"ex_{i+1}",
                "amount": amounts[i],
                "currency": "USD",
                "timestamp": 1700000000 + (i * 7200),
                "type": "WIRE",
            })
        gr.load_from_dict({"transactions": txs, "nodes": nodes})
        result = detect_layering(gr, "ex_0")
        assert result.detected is False

    def test_short_chain_not_detected(self) -> None:
        """2 hops < MIN_CHAIN_LENGTH (3)."""
        gr = GraphReasoner()
        nodes = {f"sh_{i}": {"name": f"Node {i}"} for i in range(3)}
        txs = [
            {
                "id": "TX-SHORT-1",
                "source_node": "sh_0",
                "target_node": "sh_1",
                "amount": Decimal("100000"),
                "currency": "USD",
                "timestamp": 1700000000,
                "type": "WIRE",
            },
            {
                "id": "TX-SHORT-2",
                "source_node": "sh_1",
                "target_node": "sh_2",
                "amount": Decimal("97000"),
                "currency": "USD",
                "timestamp": 1700007200,
                "type": "WIRE",
            },
        ]
        gr.load_from_dict({"transactions": txs, "nodes": nodes})
        result = detect_layering(gr, "sh_0")
        assert result.detected is False

    def test_slow_chain_not_detected(self) -> None:
        """72-hour gaps > MAX_HOP_DELAY_SECONDS (43200s = 12h)."""
        gr = GraphReasoner()
        nodes = {f"sl_{i}": {"name": f"Node {i}"} for i in range(5)}
        amounts = [Decimal("100000"), Decimal("97000"), Decimal("94090"), Decimal("91267.30")]
        txs = []
        for i in range(4):
            txs.append({
                "id": f"TX-SLOW-{i+1}",
                "source_node": f"sl_{i}",
                "target_node": f"sl_{i+1}",
                "amount": amounts[i],
                "currency": "USD",
                "timestamp": 1700000000 + (i * 259200),
                "type": "WIRE",
            })
        gr.load_from_dict({"transactions": txs, "nodes": nodes})
        result = detect_layering(gr, "sl_0")
        assert result.detected is False

    def test_increasing_amounts_not_detected(self) -> None:
        """Amounts grow along chain: not decay, not layering."""
        gr = GraphReasoner()
        nodes = {f"inc_{i}": {"name": f"Node {i}"} for i in range(5)}
        amounts = [Decimal("100000"), Decimal("103000"), Decimal("106090"), Decimal("109272.70")]
        txs = []
        for i in range(4):
            txs.append({
                "id": f"TX-INC-{i+1}",
                "source_node": f"inc_{i}",
                "target_node": f"inc_{i+1}",
                "amount": amounts[i],
                "currency": "USD",
                "timestamp": 1700000000 + (i * 7200),
                "type": "WIRE",
            })
        gr.load_from_dict({"transactions": txs, "nodes": nodes})
        result = detect_layering(gr, "inc_0")
        assert result.detected is False


# ===================================================================
# Multiple Chains Test
# ===================================================================

class TestMultipleChains:

    def test_multiple_chains_found(self, layering_boundary_scenario: dict) -> None:
        """Graph with 2 distinct valid layering paths (min2pct and max5pct)."""
        gr = _load(layering_boundary_scenario)
        chains_found = []
        for prefix in ["min2pct", "max5pct"]:
            origin = f"{prefix}_0"
            result = detect_layering(gr, origin)
            if result.detected:
                chains_found.append(prefix)
        assert len(chains_found) == 2, f"Expected 2 detected, got {chains_found}"


# ===================================================================
# v8.0 Tests
# ===================================================================

class TestV8Enhancements:

    def test_mixed_currency_chain_rejected(self) -> None:
        """v8.0 [P4v8-05]: USD->INR chain produces meaningless decay."""
        gr = GraphReasoner()
        nodes = {f"mx_{i}": {"name": f"Node {i}"} for i in range(5)}
        txs = [
            {
                "id": "TX-MX-1",
                "source_node": "mx_0",
                "target_node": "mx_1",
                "amount": Decimal("100000"),
                "currency": "USD",
                "timestamp": 1700000000,
                "type": "WIRE",
            },
            {
                "id": "TX-MX-2",
                "source_node": "mx_1",
                "target_node": "mx_2",
                "amount": Decimal("97000"),
                "currency": "INR",
                "timestamp": 1700007200,
                "type": "WIRE",
            },
            {
                "id": "TX-MX-3",
                "source_node": "mx_2",
                "target_node": "mx_3",
                "amount": Decimal("94090"),
                "currency": "USD",
                "timestamp": 1700014400,
                "type": "WIRE",
            },
            {
                "id": "TX-MX-4",
                "source_node": "mx_3",
                "target_node": "mx_4",
                "amount": Decimal("91267.30"),
                "currency": "INR",
                "timestamp": 1700021600,
                "type": "WIRE",
            },
        ]
        gr.load_from_dict({"transactions": txs, "nodes": nodes})
        result = detect_layering(gr, "mx_0")
        assert result.detected is False

    def test_chain_currency_recorded(self, layering_scenario: dict) -> None:
        """v8.0 [P4v8-08]: LayeringChain.currency is set correctly."""
        gr = _load(layering_scenario)
        result = detect_layering(gr, "layer_origin")
        assert result.detected is True
        for chain in result.chains:
            assert chain.currency == "USD"
        assert result.currency == "USD"
