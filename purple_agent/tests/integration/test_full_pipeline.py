"""
Integration Tests — Full Pipeline End-to-End
PRD Reference: Task C3
Package Version: v9.0

Tests the complete detection pipeline from graph loading through SAR generation.
Each test uses real module code (not mocks) for the graph/detection/evidence layers.
LLM calls use the mock_llm_response fixture.

v9.0: Tests for full-narrative entity hallucination, positive evidence matching,
pipe-delimited sanitization, low_confidence conditional edge, and all v8.0 tests.
v8.0: Tests for Architecture §4 Step 6 confidence formula, 8-node pipeline,
currency parameters, idempotency key, and conditional edge behavior.
"""
import hashlib
import pytest
from decimal import Decimal


@pytest.mark.integration
class TestStructuringEndToEnd:
    """Structuring detection through the full pipeline."""

    def test_structuring_detection_produces_result(self, structuring_scenario):
        """Full pipeline: load graph → detect structuring → get result."""
        from src.core.graph_reasoner import GraphReasoner
        from src.core.heuristics.structuring import detect_structuring

        reasoner = GraphReasoner()
        reasoner.load_from_dict(structuring_scenario)

        result = detect_structuring(reasoner, "mule_1")
        assert result.detected is True
        assert result.mode == "FAN_IN"
        assert result.transaction_count == 20
        assert isinstance(result.total_amount, Decimal)
        assert result.confidence > Decimal("0")

    def test_structuring_recall_100_percent(self, structuring_scenario):
        """All 20 criminal sources detected. Recall = 1.0."""
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


@pytest.mark.integration
class TestLayeringEndToEnd:
    """Layering detection through the full pipeline."""

    def test_layering_detection_produces_result(self, layering_scenario):
        """Full pipeline: load graph → detect layering → get result."""
        from src.core.graph_reasoner import GraphReasoner
        from src.core.heuristics.layering import detect_layering

        reasoner = GraphReasoner()
        reasoner.load_from_dict(layering_scenario)

        result = detect_layering(reasoner, "layer_origin")
        assert result.detected is True
        assert len(result.chains) >= 1
        assert result.confidence > Decimal("0")

    def test_layering_chain_has_decay(self, layering_scenario):
        """Detected chain has decay rates within expected range."""
        from src.core.graph_reasoner import GraphReasoner
        from src.core.heuristics.layering import detect_layering
        from src.config import DECAY_RATE_MIN, DECAY_RATE_MAX, DECAY_TOLERANCE

        reasoner = GraphReasoner()
        reasoner.load_from_dict(layering_scenario)

        result = detect_layering(reasoner, "layer_origin")
        assert result.detected is True

        for chain in result.chains:
            for rate in chain.decay_rates:
                assert DECAY_RATE_MIN - DECAY_TOLERANCE <= rate <= DECAY_RATE_MAX + DECAY_TOLERANCE

    def test_layering_currency_filter(self, layering_scenario):
        """v8.0: detect_layering with currency param filters results."""
        from src.core.graph_reasoner import GraphReasoner
        from src.core.heuristics.layering import detect_layering

        reasoner = GraphReasoner()
        reasoner.load_from_dict(layering_scenario)

        result_inr = detect_layering(reasoner, "layer_origin", currency="INR")
        assert result_inr.detected is False or len(result_inr.chains) == 0

        result_usd = detect_layering(reasoner, "layer_origin", currency="USD")
        assert result_usd.detected is True


@pytest.mark.integration
class TestCrossJurisdiction:
    """Cross-jurisdiction detection tests."""

    def test_indian_structuring_detected(self, indian_scenario):
        """INR structuring scenario detected with PMLA thresholds."""
        from src.core.graph_reasoner import GraphReasoner
        from src.core.heuristics.structuring import detect_structuring

        reasoner = GraphReasoner()
        reasoner.load_from_dict(indian_scenario)

        detected = False
        for node_id in reasoner.get_all_node_ids():
            result = detect_structuring(reasoner, node_id, currency="INR")
            if result.detected:
                detected = True
                assert result.currency == "INR"
                break

        assert detected, "Indian structuring scenario should be detected with INR thresholds"

    def test_mixed_currency_separated(self, mixed_currency_scenario):
        """v8.0: USD and INR transactions on same mule are detected separately."""
        from src.core.graph_reasoner import GraphReasoner
        from src.core.heuristics.structuring import detect_structuring

        reasoner = GraphReasoner()
        reasoner.load_from_dict(mixed_currency_scenario)

        for node_id in reasoner.get_all_node_ids():
            result_usd = detect_structuring(reasoner, node_id, currency="USD")
            result_inr = detect_structuring(reasoner, node_id, currency="INR")
            if result_usd.detected:
                assert result_usd.currency == "USD"
            if result_inr.detected:
                assert result_inr.currency == "INR"


@pytest.mark.integration
class TestConfidenceFormula:
    """v8.0 [P5v8-01/P5v8-02]: Architecture §4 Step 6 formula tests.
    v9.0 [P5v9-04]: Low confidence skip test IMPLEMENTED.
    v9.0 [P5v9-05]: Positive matching evidence boost test added.
    """

    def _make_state(self, typology="STRUCTURING", verdict="CORROBORATED", discrepancies=None):
        """Helper to construct a minimal InvestigationState for confidence tests."""
        return {
            "case_id": "TEST-CONF", "subject_id": "S1", "jurisdiction": "fincen",
            "hop_depth": 3, "graph_fragment": None, "detected_typology": typology,
            "detection_results": {}, "evidence_package": {
                "verdict": verdict,
                "discrepancies": discrepancies or [],
            },
            "sar_narrative": None, "sar_draft": None, "validation_result": None,
            "involved_entities": [], "confidence_score": 0.0,
            "investigation_timestamp": 0, "status": "IN_PROGRESS",
            "retry_count": 0, "error_message": None,
        }

    def test_confidence_formula_single_typology_with_evidence(self):
        """Single typology + corroborating evidence → 0.3 + 0.2 = 0.5 (passes gate)."""
        from src.core.decision_loop import compute_confidence
        from src.config import CONFIDENCE_THRESHOLD

        state = self._make_state(typology="STRUCTURING", verdict="CORROBORATED")
        result = compute_confidence(state)
        assert result["confidence_score"] == 0.5
        assert result["confidence_score"] >= float(CONFIDENCE_THRESHOLD)
        assert result["detected_typology"] == "STRUCTURING"

    def test_confidence_formula_single_typology_no_evidence(self):
        """Single typology without evidence → 0.3 (below threshold → NONE)."""
        from src.core.decision_loop import compute_confidence

        state = self._make_state(typology="LAYERING", verdict="INSUFFICIENT_DATA")
        result = compute_confidence(state)
        assert result["confidence_score"] == 0.3
        assert result["detected_typology"] == "NONE"

    def test_confidence_formula_both_typologies(self):
        """BOTH typologies → base 0.6 (passes gate without evidence)."""
        from src.core.decision_loop import compute_confidence

        state = self._make_state(typology="BOTH", verdict="NOT_APPLICABLE")
        result = compute_confidence(state)
        assert result["confidence_score"] == 0.6
        assert result["detected_typology"] == "BOTH"

    def test_confidence_formula_max_score(self):
        """BOTH + evidence + discrepancy → 0.6 + 0.2 + 0.2 = 1.0."""
        from src.core.decision_loop import compute_confidence

        state = self._make_state(
            typology="BOTH", verdict="CORROBORATED",
            discrepancies=["text says $15000, ledger shows $9500"],
        )
        result = compute_confidence(state)
        assert result["confidence_score"] == 1.0
        assert result["detected_typology"] == "BOTH"

    def test_confidence_formula_ieee754_exact(self):
        """All component additions are exact in IEEE 754 float64.
        v9.0 [P5v9-12]: Retained as precondition proof test.
        """
        assert 0.3 + 0.2 == 0.5
        assert 0.6 + 0.2 == 0.8
        assert 0.3 + 0.2 + 0.2 == 0.7
        assert 0.6 + 0.2 + 0.2 == 1.0

    def test_low_confidence_skips_sar(self):
        """v9.0 [P5v9-04]: Low confidence → should_generate_sar returns "submit".
        v8.0 listed this test in spec but did NOT implement it.
        Tests the conditional edge path compute_confidence → submit.
        """
        from src.core.decision_loop import compute_confidence, should_generate_sar

        state = self._make_state(typology="STRUCTURING", verdict="INSUFFICIENT_DATA")
        result = compute_confidence(state)
        assert result["confidence_score"] == 0.3
        assert result["detected_typology"] == "NONE"

        edge = should_generate_sar(result)
        assert edge == "submit", f"Expected 'submit' for low confidence, got '{edge}'"

    def test_high_confidence_generates_sar(self):
        """Complement of low confidence: high confidence → should_generate_sar returns "draft"."""
        from src.core.decision_loop import compute_confidence, should_generate_sar

        state = self._make_state(typology="BOTH", verdict="CORROBORATED")
        result = compute_confidence(state)
        assert result["confidence_score"] >= 0.5

        edge = should_generate_sar(result)
        assert edge == "draft", f"Expected 'draft' for high confidence, got '{edge}'"

    def test_evidence_boost_requires_corroboration(self):
        """v9.0 [P5v9-05]: Malformed/empty verdict does NOT trigger evidence boost."""
        from src.core.decision_loop import compute_confidence

        state_empty = self._make_state(typology="STRUCTURING", verdict="")
        result_empty = compute_confidence(state_empty)
        assert result_empty["confidence_score"] == 0.3

        state_typo = self._make_state(typology="STRUCTURING", verdict="COROBORATED")
        result_typo = compute_confidence(state_typo)
        assert result_typo["confidence_score"] == 0.3

        state_valid = self._make_state(typology="STRUCTURING", verdict="CORROBORATED")
        result_valid = compute_confidence(state_valid)
        assert result_valid["confidence_score"] == 0.5


@pytest.mark.integration
class TestIdempotencyKey:
    """v8.0 [P5v8-04]: Rule 18 idempotency key tests.
    v9.0 [P5v9-11]: Tests verify formula independently. D3 integration
    tests will cover the full submit_result path.
    """

    def test_idempotency_key_deterministic(self):
        """Same inputs produce identical SHA-256 key."""
        case_id = "CASE-001"
        typology = "STRUCTURING"
        involved = ["mule_1", "src_01", "src_02"]

        idem_input = f"{case_id}{typology}{''.join(sorted(involved))}"
        key1 = hashlib.sha256(idem_input.encode("utf-8")).hexdigest()
        key2 = hashlib.sha256(idem_input.encode("utf-8")).hexdigest()
        assert key1 == key2

    def test_idempotency_key_changes_on_typology(self):
        """Different typology → different key."""
        case_id = "CASE-001"
        involved = ["mule_1", "src_01"]

        input_a = f"{case_id}STRUCTURING{''.join(sorted(involved))}"
        input_b = f"{case_id}LAYERING{''.join(sorted(involved))}"
        key_a = hashlib.sha256(input_a.encode("utf-8")).hexdigest()
        key_b = hashlib.sha256(input_b.encode("utf-8")).hexdigest()
        assert key_a != key_b

    def test_idempotency_key_order_independent(self):
        """sorted(involved_entities) ensures order independence."""
        case_id = "CASE-001"
        typology = "BOTH"

        input_a = f"{case_id}{typology}{''.join(sorted(['z_node', 'a_node', 'm_node']))}"
        input_b = f"{case_id}{typology}{''.join(sorted(['m_node', 'a_node', 'z_node']))}"
        key_a = hashlib.sha256(input_a.encode("utf-8")).hexdigest()
        key_b = hashlib.sha256(input_b.encode("utf-8")).hexdigest()
        assert key_a == key_b


@pytest.mark.integration
class TestEmptyGraphFail:
    """v6.1 [ALN-04] + v8.0 [P4v8-03]: Empty graph handling tests."""

    def test_empty_graph_raises_on_load(self):
        """v8.0: Empty transactions list raises ValueError at B1 load."""
        from src.core.graph_reasoner import GraphReasoner

        reasoner = GraphReasoner()
        empty_data = {"nodes": {}, "transactions": []}
        with pytest.raises(ValueError):
            reasoner.load_from_dict(empty_data)


@pytest.mark.integration
class TestNaNRejection:
    """v6.1 [ALN-10]: NaN/Infinity rejection at ingestion."""

    def test_nan_amount_raises(self):
        """NaN amount must raise ValueError at ingestion."""
        from src.core.graph_reasoner import GraphReasoner

        reasoner = GraphReasoner()
        bad_data = {
            "nodes": {"A": {}, "B": {}},
            "transactions": [{
                "id": "TX-BAD",
                "source_node": "A",
                "target_node": "B",
                "amount": Decimal("NaN"),
                "currency": "USD",
                "timestamp": 1700000000,
            }]
        }
        with pytest.raises(ValueError, match="finite"):
            reasoner.load_from_dict(bad_data)

    def test_infinity_amount_raises(self):
        """Infinity amount must raise ValueError at ingestion."""
        from src.core.graph_reasoner import GraphReasoner

        reasoner = GraphReasoner()
        bad_data = {
            "nodes": {"A": {}, "B": {}},
            "transactions": [{
                "id": "TX-BAD",
                "source_node": "A",
                "target_node": "B",
                "amount": Decimal("Infinity"),
                "currency": "USD",
                "timestamp": 1700000000,
            }]
        }
        with pytest.raises(ValueError, match="finite"):
            reasoner.load_from_dict(bad_data)


@pytest.mark.integration
class TestSARValidationHallucination:
    """v9.0 [P5v9-01/P5v9-10]: Entity hallucination tests across ALL Five Ws fields."""

    def test_validate_sar_catches_entity_in_why(self, structuring_scenario):
        """v9.0: Hallucinated entity in WHY field is caught by full narrative scan."""
        from src.core.sar_drafter import SARDrafter, SARDraft

        drafter = SARDrafter()
        draft = SARDraft(
            who="mule_1, src_01",
            what="Structuring detected",
            where="US",
            when="2023-11-14",
            why="src_99 sent $9500 to mule_1 via structuring pattern",
            cited_tx_ids=["TX-STRUCT-001"],
            raw_narrative="<WHO>mule_1, src_01</WHO><WHY>src_99 sent $9500 to mule_1</WHY>",
            jurisdiction="fincen",
        )
        validation = drafter.validate_sar(draft, structuring_scenario)
        assert validation.passed is False
        assert any("src_99" in e for e in validation.errors), (
            f"Expected hallucination error for src_99. Got: {validation.errors}"
        )

    def test_validate_sar_catches_entity_in_what(self, structuring_scenario):
        """v9.0: Hallucinated entity in WHAT field is caught."""
        from src.core.sar_drafter import SARDrafter, SARDraft

        drafter = SARDrafter()
        draft = SARDraft(
            who="mule_1",
            what="mule_99 participated in structuring",
            where="US",
            when="2023-11-14",
            why="Pattern detected",
            cited_tx_ids=["TX-STRUCT-001"],
            raw_narrative="<WHO>mule_1</WHO><WHAT>mule_99 participated</WHAT>",
            jurisdiction="fincen",
        )
        validation = drafter.validate_sar(draft, structuring_scenario)
        assert validation.passed is False
        assert any("mule_99" in e for e in validation.errors), (
            f"Expected hallucination error for mule_99. Got: {validation.errors}"
        )


@pytest.mark.integration
class TestSanitizer:
    """v9.0 [P5v9-07]: Extended sanitizer tests."""

    def test_sanitize_strips_pipe_delimiters(self):
        """v9.0: Pipe-delimited data tags are stripped."""
        from src.core.sar_drafter import _sanitize_for_prompt

        assert "[DATA_TAG_STRIPPED]" in _sanitize_for_prompt("prefix <data> suffix")
        assert "[DATA_TAG_STRIPPED]" in _sanitize_for_prompt("prefix </data> suffix")

        assert "[DATA_TAG_STRIPPED]" in _sanitize_for_prompt("prefix <|data> suffix")
        assert "[DATA_TAG_STRIPPED]" in _sanitize_for_prompt("prefix <|data|> suffix")
        assert "[DATA_TAG_STRIPPED]" in _sanitize_for_prompt("prefix <|/data> suffix")
        assert "[DATA_TAG_STRIPPED]" in _sanitize_for_prompt("prefix <|/data|> suffix")
