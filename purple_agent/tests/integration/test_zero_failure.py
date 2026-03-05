"""
Zero-Failure Integration Tests — Determinism + Hallucination + Recall
PRD Reference: Task C3
Package Version: v9.0

These tests enforce the Zero-Failure Mandate:
1. Determinism: 10 identical runs produce byte-identical results
2. Zero Hallucination: No entity/TX/amount not in the graph
3. 100% Recall: All criminal nodes detected
4. v9.0: Full-narrative entity validation, positive evidence matching
5. v8.0: Architecture §4 Step 6 formula, idempotency, currency filtering
"""
import hashlib
import pytest
from decimal import Decimal


@pytest.mark.integration
class TestDeterminism:
    """Every detection must be perfectly reproducible."""

    def test_graph_detection_determinism(self, structuring_scenario):
        """10 runs of graph load + detection must produce identical results."""
        from src.core.graph_reasoner import GraphReasoner
        from src.core.heuristics.structuring import detect_structuring

        results: list[dict] = []

        for _ in range(10):
            reasoner = GraphReasoner()
            reasoner.load_from_dict(structuring_scenario)

            run_result = {}
            for node_id in reasoner.get_all_node_ids():
                detection = detect_structuring(reasoner, node_id)
                if detection.detected:
                    run_result[node_id] = {
                        "mode": detection.mode,
                        "sources": detection.counterpart_nodes,
                        "count": detection.transaction_count,
                        "total": str(detection.total_amount),
                        "confidence": str(detection.confidence),
                    }
            results.append(run_result)

        for i in range(1, 10):
            assert results[i] == results[0], (
                f"Run {i} differs from run 0.\n"
                f"Run 0: {results[0]}\n"
                f"Run {i}: {results[i]}"
            )

    def test_sorted_node_iteration_across_runs(self, structuring_scenario):
        """Verify get_all_node_ids returns identical order across instantiations."""
        from src.core.graph_reasoner import GraphReasoner

        orders: list[list[str]] = []
        for _ in range(5):
            r = GraphReasoner()
            r.load_from_dict(structuring_scenario)
            orders.append(r.get_all_node_ids())

        for i in range(1, 5):
            assert orders[i] == orders[0]


@pytest.mark.integration
class TestZeroHallucination:
    """Verify no hallucinated entities appear in results."""

    def test_no_hallucinated_entities_in_structuring(self, structuring_scenario):
        """Every entity in StructuringResult.counterpart_nodes must exist in the graph."""
        from src.core.graph_reasoner import GraphReasoner
        from src.core.heuristics.structuring import detect_structuring

        reasoner = GraphReasoner()
        reasoner.load_from_dict(structuring_scenario)
        graph_nodes = set(reasoner.get_all_node_ids())

        for node_id in reasoner.get_all_node_ids():
            result = detect_structuring(reasoner, node_id)
            if result.detected:
                for source in result.counterpart_nodes:
                    assert source in graph_nodes, (
                        f"Hallucinated entity: {source} not in graph"
                    )

    def test_no_hallucinated_entities_in_layering(self, layering_scenario):
        """Every node in LayeringResult chains must exist in the graph."""
        from src.core.graph_reasoner import GraphReasoner
        from src.core.heuristics.layering import detect_layering

        reasoner = GraphReasoner()
        reasoner.load_from_dict(layering_scenario)
        graph_nodes = set(reasoner.get_all_node_ids())

        for node_id in reasoner.get_all_node_ids():
            result = detect_layering(reasoner, node_id)
            if result.detected:
                for chain in result.chains:
                    for chain_node in chain.chain_nodes:
                        assert chain_node in graph_nodes, (
                            f"Hallucinated entity: {chain_node} not in graph"
                        )

    def test_no_hallucinated_transactions_in_sar(self, structuring_scenario):
        """SAR validation catches hallucinated TX IDs."""
        from src.core.sar_drafter import SARDrafter, SARDraft

        drafter = SARDrafter()
        draft = SARDraft(
            who="mule_1", what="Structuring", where="US",
            when="2023-11-14", why="Pattern detected TX-FAKE-999 ($9500)",
            cited_tx_ids=["TX-STRUCT-001", "TX-FAKE-999"],
            raw_narrative="Investigation found TX-FAKE-999 ($9500) suspicious",
            jurisdiction="fincen",
        )
        validation = drafter.validate_sar(draft, structuring_scenario)
        assert validation.passed is False
        assert any("TX-FAKE-999" in e for e in validation.errors)


@pytest.mark.integration
class TestRecall:
    """Verify 100% recall on criminal nodes."""

    def test_recall_100_percent_structuring(self, structuring_scenario):
        """All 20 injected structuring source nodes detected. Recall = 1.0."""
        from src.core.graph_reasoner import GraphReasoner
        from src.core.heuristics.structuring import detect_structuring

        reasoner = GraphReasoner()
        reasoner.load_from_dict(structuring_scenario)

        expected_criminals = set(f"src_{i:02d}" for i in range(1, 21))
        detected_criminals: set[str] = set()

        for node_id in reasoner.get_all_node_ids():
            result = detect_structuring(reasoner, node_id)
            if result.detected:
                detected_criminals.update(result.counterpart_nodes)

        true_positives = detected_criminals & expected_criminals
        recall = len(true_positives) / len(expected_criminals)
        assert recall == 1.0, (
            f"Recall = {recall}. Missing: {expected_criminals - detected_criminals}"
        )

    def test_multidiraph_preserves_all_edges(self, multi_edge_scenario):
        """3 parallel edges A→B preserved in MultiDiGraph."""
        from src.core.graph_reasoner import GraphReasoner

        reasoner = GraphReasoner()
        reasoner.load_from_dict(multi_edge_scenario)

        assert reasoner.get_edge_count() == 3
        outgoing = reasoner.get_1hop_outgoing("A")
        assert len(outgoing) == 3

    def test_confidence_score_nonzero_on_detection(self, structuring_scenario):
        """confidence_score > 0 when structuring is detected."""
        from src.core.graph_reasoner import GraphReasoner
        from src.core.heuristics.structuring import detect_structuring

        reasoner = GraphReasoner()
        reasoner.load_from_dict(structuring_scenario)

        result = detect_structuring(reasoner, "mule_1")
        assert result.detected is True
        assert result.confidence > Decimal("0"), "confidence_score must be > 0 on detection"


@pytest.mark.integration
class TestMechanicalSAR:
    """v6.1 [ALN-05]: Mechanical SAR template tests."""

    def test_mechanical_template_produces_valid_five_ws(self, structuring_scenario):
        """Mechanical template must produce all Five Ws sections."""
        from src.core.sar_drafter import mechanical_sar_template

        detection_results = {
            "typology": "STRUCTURING",
            "structuring_hits": [{
                "node": "mule_1",
                "mode": "FAN_IN",
                "sources": ["src_01", "src_02"],
                "tx_count": 2,
                "total": "18500",
                "currency": "USD",
                "transactions": ["TX-STRUCT-001", "TX-STRUCT-002"],
            }],
            "layering_hits": [],
        }

        draft = mechanical_sar_template(detection_results, structuring_scenario, "fincen")
        assert draft.who != ""
        assert draft.what != ""
        assert draft.where != ""
        assert draft.when != ""
        assert draft.why != ""
        assert len(draft.cited_tx_ids) > 0

    def test_mechanical_sar_iso_timestamps(self, structuring_scenario):
        """v8.0 [P5v8-11]: Mechanical SAR uses ISO 8601, not raw epoch."""
        from src.core.sar_drafter import mechanical_sar_template

        detection_results = {
            "typology": "STRUCTURING",
            "structuring_hits": [{
                "node": "mule_1", "mode": "FAN_IN",
                "sources": ["src_01"], "tx_count": 1,
                "total": "9200", "currency": "USD",
                "transactions": ["TX-STRUCT-001"],
            }],
            "layering_hits": [],
        }

        draft = mechanical_sar_template(detection_results, structuring_scenario, "fincen")
        assert "T" in draft.when or draft.when == "Timestamps unavailable"
        if draft.when != "Timestamps unavailable":
            assert not draft.when.startswith("Timestamp range: 1")

    def test_mechanical_template_contains_only_graph_entities(self, structuring_scenario):
        """Mechanical template references only entities from the graph."""
        from src.core.sar_drafter import mechanical_sar_template
        from src.core.graph_reasoner import GraphReasoner

        reasoner = GraphReasoner()
        reasoner.load_from_dict(structuring_scenario)
        graph_nodes = set(reasoner.get_all_node_ids())

        detection_results = {
            "typology": "STRUCTURING",
            "structuring_hits": [{
                "node": "mule_1", "mode": "FAN_IN",
                "sources": ["src_01"], "tx_count": 1,
                "total": "9200", "currency": "USD",
                "transactions": ["TX-STRUCT-001"],
            }],
            "layering_hits": [],
        }

        draft = mechanical_sar_template(detection_results, structuring_scenario, "fincen")
        for entity in draft.who.split(", "):
            entity = entity.strip()
            if entity and entity != "Entities not identified":
                assert entity in graph_nodes, f"Mechanical SAR references non-graph entity: {entity}"
