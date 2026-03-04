"""
Tests for Structuring (Smurfing) Detection -- Purple Agent v8.0
PRD Reference: Task B2
TDD: Tests written FIRST, implementation SECOND.

Tests cover:
  - Fan-in smurfing detection (20 sub-threshold deposits)
  - Counterpart node identification (100% recall)
  - Correct mule identification
  - Transaction count and totals (Decimal)
  - Confidence scoring (valid range, increases with more TXs)
  - Negative cases (legitimate hub, below/above threshold, too few TXs)
  - Boundary testing (exact min $9,000, exact max $9,800)
  - INR jurisdiction (Indian PMLA thresholds)
  - Fan-out detection and fan-in precedence
  - Time window expiration
  - v8.0: Mixed currency isolation (Rule 14)
"""
from decimal import Decimal

import pytest

from src.core.graph_reasoner import GraphReasoner
from src.core.heuristics.structuring import detect_structuring, StructuringResult
from src.config import STRUCTURING_MIN_COUNT


# ===================================================================
# Helper to load scenario into GraphReasoner
# ===================================================================

def _load(scenario: dict) -> GraphReasoner:
    gr = GraphReasoner()
    gr.load_from_dict(scenario)
    return gr


# ===================================================================
# Detection Tests (Fan-In)
# ===================================================================

class TestFanInDetection:

    def test_detects_smurfing_fan_in(self, structuring_scenario: dict) -> None:
        gr = _load(structuring_scenario)
        result = detect_structuring(gr, "mule_1", "USD")
        assert result.detected is True
        assert result.mode == "FAN_IN"

    def test_identifies_all_counterpart_nodes(self, structuring_scenario: dict) -> None:
        gr = _load(structuring_scenario)
        result = detect_structuring(gr, "mule_1", "USD")
        expected = sorted(structuring_scenario["criminal_sources"])
        assert result.counterpart_nodes == expected

    def test_correct_mule_node(self, structuring_scenario: dict) -> None:
        gr = _load(structuring_scenario)
        result = detect_structuring(gr, "mule_1", "USD")
        assert result.target_node == "mule_1"

    def test_transaction_count(self, structuring_scenario: dict) -> None:
        gr = _load(structuring_scenario)
        result = detect_structuring(gr, "mule_1", "USD")
        assert result.transaction_count == 20

    def test_total_amount_is_decimal(self, structuring_scenario: dict) -> None:
        gr = _load(structuring_scenario)
        result = detect_structuring(gr, "mule_1", "USD")
        if not isinstance(result.total_amount, Decimal):
            raise TypeError(
                f"total_amount is {type(result.total_amount).__name__}, expected Decimal"
            )
        assert result.total_amount == structuring_scenario["expected_total_criminal_inflow"]

    def test_all_transactions_have_ids(self, structuring_scenario: dict) -> None:
        gr = _load(structuring_scenario)
        result = detect_structuring(gr, "mule_1", "USD")
        for tx in result.qualifying_transactions:
            assert tx.get("tx_id"), f"Transaction missing tx_id: {tx}"


# ===================================================================
# Confidence Tests
# ===================================================================

class TestConfidence:

    def test_confidence_score_in_valid_range(self, structuring_scenario: dict) -> None:
        gr = _load(structuring_scenario)
        result = detect_structuring(gr, "mule_1", "USD")
        assert Decimal("0.33") <= result.confidence <= Decimal("1")

    def test_confidence_increases_with_more_transactions(self) -> None:
        """30 TXs should yield higher confidence than 10 TXs (at min_count boundary)."""
        def _build(n: int) -> GraphReasoner:
            gr = GraphReasoner()
            nodes = {"target": {"name": "Target"}}
            txs = []
            for i in range(1, n + 1):
                src = f"src_{i:03d}"
                nodes[src] = {"name": f"Source {i}"}
                txs.append({
                    "id": f"TX-CONF-{i:03d}",
                    "source_node": src,
                    "target_node": "target",
                    "amount": Decimal("9200"),
                    "currency": "USD",
                    "timestamp": 1700000000 + (i * 600),
                    "type": "WIRE",
                })
            gr.load_from_dict({"transactions": txs, "nodes": nodes})
            return gr

        gr_10 = _build(STRUCTURING_MIN_COUNT)
        gr_30 = _build(30)
        result_10 = detect_structuring(gr_10, "target", "USD")
        result_30 = detect_structuring(gr_30, "target", "USD")
        assert result_10.detected is True
        assert result_30.detected is True
        assert result_30.confidence > result_10.confidence


# ===================================================================
# Negative Cases
# ===================================================================

class TestNegativeCases:

    def test_legitimate_hub_not_detected(self) -> None:
        """Clean fixture: 5 salary deposits of $3,000 each. Not structuring."""
        gr = GraphReasoner()
        nodes = {"corp": {"name": "LegitCorp"}}
        txs = []
        for i in range(1, 6):
            src = f"emp_{i}"
            nodes[src] = {"name": f"Employee {i}"}
            txs.append({
                "id": f"TX-SAL-{i}",
                "source_node": src,
                "target_node": "corp",
                "amount": Decimal("3000"),
                "currency": "USD",
                "timestamp": 1700000000 + (i * 3600),
                "type": "WIRE",
            })
        gr.load_from_dict({"transactions": txs, "nodes": nodes})
        result = detect_structuring(gr, "corp", "USD")
        assert result.detected is False

    def test_below_threshold_not_detected(self) -> None:
        """$8,999.99 amounts: below $9K structuring range."""
        gr = GraphReasoner()
        nodes = {"target": {"name": "Target"}}
        txs = []
        for i in range(1, 21):
            src = f"src_{i:02d}"
            nodes[src] = {"name": f"Source {i}"}
            txs.append({
                "id": f"TX-BELOW-{i:03d}",
                "source_node": src,
                "target_node": "target",
                "amount": Decimal("8999.99"),
                "currency": "USD",
                "timestamp": 1700000000 + (i * 1800),
                "type": "WIRE",
            })
        gr.load_from_dict({"transactions": txs, "nodes": nodes})
        result = detect_structuring(gr, "target", "USD")
        assert result.detected is False

    def test_above_threshold_not_detected(self) -> None:
        """$10,001 amounts: above CTR, not structuring band."""
        gr = GraphReasoner()
        nodes = {"target": {"name": "Target"}}
        txs = []
        for i in range(1, 21):
            src = f"src_{i:02d}"
            nodes[src] = {"name": f"Source {i}"}
            txs.append({
                "id": f"TX-ABOVE-{i:03d}",
                "source_node": src,
                "target_node": "target",
                "amount": Decimal("10001"),
                "currency": "USD",
                "timestamp": 1700000000 + (i * 1800),
                "type": "WIRE",
            })
        gr.load_from_dict({"transactions": txs, "nodes": nodes})
        result = detect_structuring(gr, "target", "USD")
        assert result.detected is False

    def test_too_few_transactions_not_detected(self) -> None:
        """5 transactions < STRUCTURING_MIN_COUNT (10)."""
        gr = GraphReasoner()
        nodes = {"target": {"name": "Target"}}
        txs = []
        for i in range(1, 6):
            src = f"src_{i}"
            nodes[src] = {"name": f"Source {i}"}
            txs.append({
                "id": f"TX-FEW-{i}",
                "source_node": src,
                "target_node": "target",
                "amount": Decimal("9200"),
                "currency": "USD",
                "timestamp": 1700000000 + (i * 1800),
                "type": "WIRE",
            })
        gr.load_from_dict({"transactions": txs, "nodes": nodes})
        result = detect_structuring(gr, "target", "USD")
        assert result.detected is False


# ===================================================================
# Boundary Tests
# ===================================================================

class TestBoundaries:

    def test_boundary_exact_min_amount(self) -> None:
        """$9,000.00 exact: included in detection range (>=)."""
        gr = GraphReasoner()
        nodes = {"target": {"name": "Target"}}
        txs = []
        for i in range(1, STRUCTURING_MIN_COUNT + 1):
            src = f"src_{i:02d}"
            nodes[src] = {"name": f"Source {i}"}
            txs.append({
                "id": f"TX-EXMIN-{i:03d}",
                "source_node": src,
                "target_node": "target",
                "amount": Decimal("9000"),
                "currency": "USD",
                "timestamp": 1700000000 + (i * 1800),
                "type": "WIRE",
            })
        gr.load_from_dict({"transactions": txs, "nodes": nodes})
        result = detect_structuring(gr, "target", "USD")
        assert result.detected is True

    def test_boundary_exact_max_amount(self) -> None:
        """$9,800.00 exact: included in detection range (<=)."""
        gr = GraphReasoner()
        nodes = {"target": {"name": "Target"}}
        txs = []
        for i in range(1, STRUCTURING_MIN_COUNT + 1):
            src = f"src_{i:02d}"
            nodes[src] = {"name": f"Source {i}"}
            txs.append({
                "id": f"TX-EXMAX-{i:03d}",
                "source_node": src,
                "target_node": "target",
                "amount": Decimal("9800"),
                "currency": "USD",
                "timestamp": 1700000000 + (i * 1800),
                "type": "WIRE",
            })
        gr.load_from_dict({"transactions": txs, "nodes": nodes})
        result = detect_structuring(gr, "target", "USD")
        assert result.detected is True


# ===================================================================
# INR Jurisdiction Tests
# ===================================================================

class TestINRJurisdiction:

    def test_indian_structuring_detected(self, indian_scenario: dict) -> None:
        gr = _load(indian_scenario)
        result = detect_structuring(gr, "mule_in_1", "INR")
        assert result.detected is True
        assert result.mode == "FAN_IN"
        assert result.currency == "INR"


# ===================================================================
# Fan-Out and Precedence Tests
# ===================================================================

class TestFanOutAndPrecedence:

    def test_fan_out_detected(self) -> None:
        """One source, many sub-threshold outgoing -> FAN_OUT detected."""
        gr = GraphReasoner()
        nodes = {"source": {"name": "Source"}}
        txs = []
        for i in range(1, STRUCTURING_MIN_COUNT + 1):
            tgt = f"tgt_{i:02d}"
            nodes[tgt] = {"name": f"Target {i}"}
            txs.append({
                "id": f"TX-FOUT-{i:03d}",
                "source_node": "source",
                "target_node": tgt,
                "amount": Decimal("9300"),
                "currency": "USD",
                "timestamp": 1700000000 + (i * 1800),
                "type": "WIRE",
            })
        gr.load_from_dict({"transactions": txs, "nodes": nodes})
        result = detect_structuring(gr, "source", "USD")
        assert result.detected is True
        assert result.mode == "FAN_OUT"

    def test_fan_in_takes_precedence_over_fan_out(self) -> None:
        """v7.0 [P4-17]: Node with BOTH incoming and outgoing structuring -> FAN_IN returned."""
        gr = GraphReasoner()
        nodes = {"hub": {"name": "Hub"}}
        txs = []
        for i in range(1, STRUCTURING_MIN_COUNT + 1):
            src = f"in_{i:02d}"
            nodes[src] = {"name": f"In Source {i}"}
            txs.append({
                "id": f"TX-HUB-IN-{i:03d}",
                "source_node": src,
                "target_node": "hub",
                "amount": Decimal("9400"),
                "currency": "USD",
                "timestamp": 1700000000 + (i * 1800),
                "type": "WIRE",
            })
        for i in range(1, STRUCTURING_MIN_COUNT + 1):
            tgt = f"out_{i:02d}"
            nodes[tgt] = {"name": f"Out Target {i}"}
            txs.append({
                "id": f"TX-HUB-OUT-{i:03d}",
                "source_node": "hub",
                "target_node": tgt,
                "amount": Decimal("9500"),
                "currency": "USD",
                "timestamp": 1700000000 + 50000 + (i * 1800),
                "type": "WIRE",
            })
        gr.load_from_dict({"transactions": txs, "nodes": nodes})
        result = detect_structuring(gr, "hub", "USD")
        assert result.detected is True
        assert result.mode == "FAN_IN"


# ===================================================================
# Time Window Tests
# ===================================================================

class TestTimeWindow:

    def test_time_window_expired_not_detected(self) -> None:
        """Transactions >48hrs apart: outside STRUCTURING_TIME_WINDOW_SECONDS."""
        gr = GraphReasoner()
        nodes = {"target": {"name": "Target"}}
        txs = []
        for i in range(1, STRUCTURING_MIN_COUNT + 1):
            src = f"src_{i:02d}"
            nodes[src] = {"name": f"Source {i}"}
            txs.append({
                "id": f"TX-EXPIRED-{i:03d}",
                "source_node": src,
                "target_node": "target",
                "amount": Decimal("9200"),
                "currency": "USD",
                "timestamp": 1700000000 + (i * 200000),
                "type": "WIRE",
            })
        gr.load_from_dict({"transactions": txs, "nodes": nodes})
        result = detect_structuring(gr, "target", "USD")
        assert result.detected is False


# ===================================================================
# v8.0 Mixed Currency Tests
# ===================================================================

class TestMixedCurrency:

    def test_mixed_currency_not_cross_detected(self, mixed_currency_scenario: dict) -> None:
        """v8.0 [P4v8-11]: INR TXs don't trigger USD detection and vice versa.

        Verifies that when detecting USD structuring, only USD transactions
        are considered in the qualifying count and amounts.
        """
        gr = _load(mixed_currency_scenario)

        usd_result = detect_structuring(gr, "mixed_mule", "USD")
        assert usd_result.detected is True
        assert usd_result.currency == "USD"
        for tx in usd_result.qualifying_transactions:
            assert tx["currency"] == "USD", (
                f"Non-USD transaction {tx['tx_id']} found in USD detection result"
            )

        inr_result = detect_structuring(gr, "mixed_mule", "INR")
        assert inr_result.detected is True
        assert inr_result.currency == "INR"
        for tx in inr_result.qualifying_transactions:
            assert tx["currency"] == "INR", (
                f"Non-INR transaction {tx['tx_id']} found in INR detection result"
            )
