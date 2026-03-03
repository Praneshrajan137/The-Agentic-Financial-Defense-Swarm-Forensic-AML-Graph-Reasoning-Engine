"""
Test: LangGraph Decision Loop -- State Machine
PRD Reference: Task A7
"""
import pytest
from decimal import Decimal
from typing import Any

from src.core.decision_loop import (
    InvestigationState,
    ingest,
    detect_structuring,
    detect_layering,
    synthesize_evidence,
    compute_confidence,
    should_generate_sar,
    draft_sar,
    validate_sar,
    should_retry,
    submit,
    build_workflow,
)


def _make_initial_state(**overrides: Any) -> InvestigationState:
    """Build a minimal valid initial state with optional overrides."""
    base: InvestigationState = {
        "case_id": "CASE-TEST-001",
        "subject_id": "suspect_001",
        "jurisdiction": "fincen",
        "hop_depth": 3,
        "graph_fragment": None,
        "transactions": [],
        "nodes": {},
        "text_evidence": [],
        "ground_truth_criminals": [],
        "structuring_results": [],
        "layering_results": [],
        "detected_typology": None,
        "involved_entities": [],
        "evidence_results": [],
        "evidence_package": None,
        "sar_narrative": None,
        "validation_result": None,
        "confidence_score": 0.0,
        "investigation_start_timestamp": 0,
        "investigation_timestamp": 0,
        "status": "pending",
        "retry_count": 0,
        "error_message": None,
    }
    return {**base, **overrides}


# ═══════════════════════════════════════════════════════════════════
# Ingest Node
# ═══════════════════════════════════════════════════════════════════

class TestIngest:
    def test_ingest_fails_on_none_graph(self):
        state = _make_initial_state(graph_fragment=None)
        result = ingest(state)
        if result["status"] != "failed":
            raise ValueError("Should fail on None graph_fragment")
        if result["error_message"] is None:
            raise ValueError("Error message should be set")

    def test_ingest_extracts_from_graph_fragment(self):
        fragment = {
            "transactions": [{"id": "TX-1"}],
            "nodes": {"a": {"id": "a"}},
            "text_evidence": [{"id": "EV-1"}],
            "ground_truth_criminals": ["c1"],
        }
        state = _make_initial_state(graph_fragment=fragment)
        result = ingest(state)
        if result["status"] != "in_progress":
            raise ValueError(f"Expected in_progress, got {result['status']}")
        if len(result["transactions"]) != 1:
            raise ValueError("Transactions not extracted")
        if "a" not in result["nodes"]:
            raise ValueError("Nodes not extracted")
        if result["investigation_start_timestamp"] == 0:
            raise ValueError("Start timestamp not set")


# ═══════════════════════════════════════════════════════════════════
# Detection Nodes (placeholders)
# ═══════════════════════════════════════════════════════════════════

class TestDetection:
    def test_detect_structuring_passes_through_on_failure(self):
        state = _make_initial_state(status="failed")
        result = detect_structuring(state)
        if result["status"] != "failed":
            raise ValueError("Should preserve failed status")

    def test_detect_structuring_returns_empty_results(self):
        state = _make_initial_state(status="in_progress")
        result = detect_structuring(state)
        if result["structuring_results"] != []:
            raise ValueError("Phase A should return empty results")

    def test_detect_layering_passes_through_on_failure(self):
        state = _make_initial_state(status="failed")
        result = detect_layering(state)
        if result["status"] != "failed":
            raise ValueError("Should preserve failed status")

    def test_detect_layering_returns_empty_results(self):
        state = _make_initial_state(status="in_progress")
        result = detect_layering(state)
        if result["layering_results"] != []:
            raise ValueError("Phase A should return empty results")


# ═══════════════════════════════════════════════════════════════════
# Evidence Synthesis
# ═══════════════════════════════════════════════════════════════════

class TestEvidenceSynthesis:
    def test_synthesize_evidence_returns_package(self):
        state = _make_initial_state(status="in_progress")
        result = synthesize_evidence(state)
        pkg = result["evidence_package"]
        if pkg is None:
            raise ValueError("evidence_package should not be None")
        if pkg["text_corroborates_ledger"] is not False:
            raise ValueError("Phase A should return False for corroboration")
        if pkg["has_suspicious_discrepancy"] is not False:
            raise ValueError("Phase A should return False for discrepancy")

    def test_synthesize_evidence_skips_on_failure(self):
        state = _make_initial_state(status="failed")
        result = synthesize_evidence(state)
        if result["status"] != "failed":
            raise ValueError("Should preserve failed status")


# ═══════════════════════════════════════════════════════════════════
# Confidence Scoring (uses state_builder from conftest v12.0)
# ═══════════════════════════════════════════════════════════════════

class TestConfidenceScoring:
    def test_confidence_all_cases(self, confidence_scoring_scenario):
        """Test all 5 confidence scoring cases using state_builder (MED-31)."""
        initial = _make_initial_state(status="in_progress")

        for name, case in confidence_scoring_scenario["test_cases"].items():
            state = {**initial, **case["state_builder"]}
            result = compute_confidence(state)
            actual = result["confidence_score"]
            expected = case["expected_score"]
            if actual != expected:
                raise ValueError(
                    f"Case '{name}': expected {expected}, got {actual}"
                )

    def test_confidence_high_generates_sar(self, confidence_scoring_scenario):
        """High confidence should route to SAR generation."""
        initial = _make_initial_state(status="in_progress")
        case = confidence_scoring_scenario["test_cases"]["high_confidence"]
        state = {**initial, **case["state_builder"]}
        result = compute_confidence(state)
        route = should_generate_sar(result)
        if route != "draft":
            raise ValueError(f"Expected 'draft', got '{route}'")

    def test_confidence_low_skips_sar(self, confidence_scoring_scenario):
        """Low confidence should skip SAR and go to submit."""
        initial = _make_initial_state(status="in_progress")
        case = confidence_scoring_scenario["test_cases"]["low_confidence"]
        state = {**initial, **case["state_builder"]}
        result = compute_confidence(state)
        route = should_generate_sar(result)
        if route != "submit":
            raise ValueError(f"Expected 'submit', got '{route}'")

    def test_confidence_zero_clears_entities(self, confidence_scoring_scenario):
        """Zero confidence should clear involved_entities (MED-22)."""
        initial = _make_initial_state(status="in_progress")
        case = confidence_scoring_scenario["test_cases"]["zero_confidence"]
        state = {**initial, **case["state_builder"]}
        result = compute_confidence(state)
        if result["detected_typology"] != "NONE":
            raise ValueError("Should be NONE")
        if result["involved_entities"] != []:
            raise ValueError("Should clear entities on zero confidence")

    def test_confidence_exact_boundary(self, confidence_scoring_scenario):
        """Exact boundary (0.5) should generate SAR (not < threshold)."""
        initial = _make_initial_state(status="in_progress")
        case = confidence_scoring_scenario["test_cases"]["exact_boundary"]
        state = {**initial, **case["state_builder"]}
        result = compute_confidence(state)
        if result["confidence_score"] != 0.5:
            raise ValueError(f"Expected 0.5, got {result['confidence_score']}")
        route = should_generate_sar(result)
        if route != "draft":
            raise ValueError("0.5 == threshold; should generate SAR")

    def test_confidence_failed_state_passthrough(self):
        """Failed state should pass through unchanged."""
        state = _make_initial_state(status="failed")
        result = compute_confidence(state)
        if result["status"] != "failed":
            raise ValueError("Should preserve failed status")


# ═══════════════════════════════════════════════════════════════════
# SAR Drafting and Validation
# ═══════════════════════════════════════════════════════════════════

class TestSARDraftingAndValidation:
    def test_draft_sar_sets_narrative(self):
        state = _make_initial_state(status="in_progress", sar_narrative=None)
        result = draft_sar(state)
        if result["sar_narrative"] is None:
            raise ValueError("SAR narrative should be set")

    def test_validate_sar_increments_retry_count(self):
        state = _make_initial_state(
            status="in_progress",
            sar_narrative="Test narrative",
            retry_count=0,
        )
        result = validate_sar(state)
        if result["retry_count"] != 1:
            raise ValueError(f"Expected retry_count=1, got {result['retry_count']}")

    def test_validate_sar_passes_with_valid_narrative(self):
        state = _make_initial_state(
            status="in_progress",
            sar_narrative="Investigation found suspect_001 involved in structuring",
            involved_entities=["suspect_001"],
            retry_count=0,
        )
        result = validate_sar(state)
        if not result["validation_result"]["passed"]:
            raise ValueError("Valid narrative with referenced entity should pass")

    def test_validate_sar_fails_on_empty_narrative(self):
        state = _make_initial_state(
            status="in_progress",
            sar_narrative="",
            retry_count=0,
        )
        result = validate_sar(state)
        if result["validation_result"]["passed"]:
            raise ValueError("Empty narrative should fail validation")
        if result["sar_narrative"] is not None:
            raise ValueError("Failed validation should clear sar_narrative to None")
        if "empty or missing" not in result["validation_result"]["errors"][0]:
            raise ValueError("Error should mention empty/missing narrative")

    def test_validate_sar_fails_on_none_narrative(self):
        state = _make_initial_state(
            status="in_progress",
            sar_narrative=None,
            retry_count=0,
        )
        result = validate_sar(state)
        if result["validation_result"]["passed"]:
            raise ValueError("None narrative should fail validation")
        if result["sar_narrative"] is not None:
            raise ValueError("Failed validation should clear sar_narrative to None")

    def test_validate_sar_fails_when_no_entity_referenced(self):
        state = _make_initial_state(
            status="in_progress",
            sar_narrative="Generic narrative with no specific entity names",
            involved_entities=["suspect_001", "mule_1"],
            retry_count=0,
        )
        result = validate_sar(state)
        if result["validation_result"]["passed"]:
            raise ValueError("Narrative without any involved entity should fail")
        if result["sar_narrative"] is not None:
            raise ValueError("Failed validation should clear sar_narrative to None")

    def test_validate_sar_passes_with_no_entities(self):
        """If involved_entities is empty, entity-reference check is skipped."""
        state = _make_initial_state(
            status="in_progress",
            sar_narrative="Some narrative text",
            involved_entities=[],
            retry_count=0,
        )
        result = validate_sar(state)
        if not result["validation_result"]["passed"]:
            raise ValueError("Narrative with no entities to check should pass")

    def test_validate_sar_failure_triggers_retry_flow(self):
        """Integration: validate_sar failure -> should_retry -> draft."""
        state = _make_initial_state(
            status="in_progress",
            sar_narrative="",
            retry_count=0,
        )
        validated = validate_sar(state)
        decision = should_retry(validated)
        if decision != "draft":
            raise ValueError(f"Expected retry to 'draft', got '{decision}'")

    def test_should_retry_submit_on_pass(self):
        state = _make_initial_state(
            validation_result={"passed": True, "errors": []},
            retry_count=1,
        )
        if should_retry(state) != "submit":
            raise ValueError("Passed validation should submit")

    def test_should_retry_draft_on_failure(self):
        state = _make_initial_state(
            validation_result={"passed": False, "errors": ["err"]},
            retry_count=1,
        )
        if should_retry(state) != "draft":
            raise ValueError("Failed validation should retry draft")

    def test_should_retry_submit_on_max_retries(self):
        state = _make_initial_state(
            validation_result={"passed": False, "errors": ["err"]},
            retry_count=3,
        )
        if should_retry(state) != "submit":
            raise ValueError("Max retries should submit")


# ═══════════════════════════════════════════════════════════════════
# Submit Node
# ═══════════════════════════════════════════════════════════════════

class TestSubmit:
    def test_submit_sets_completed_status(self):
        state = _make_initial_state(status="in_progress")
        result = submit(state)
        if result["status"] != "completed":
            raise ValueError(f"Expected completed, got {result['status']}")

    def test_submit_preserves_failed_status(self):
        state = _make_initial_state(status="failed")
        result = submit(state)
        if result["status"] != "failed":
            raise ValueError("Should preserve failed status")

    def test_submit_sets_investigation_timestamp(self):
        state = _make_initial_state(status="in_progress")
        result = submit(state)
        if result["investigation_timestamp"] == 0:
            raise ValueError("Timestamp should be set")


# ═══════════════════════════════════════════════════════════════════
# Workflow Compilation and End-to-End
# ═══════════════════════════════════════════════════════════════════

class TestWorkflow:
    def test_workflow_compiles(self):
        workflow = build_workflow()
        if workflow is None:
            raise ValueError("Workflow should compile successfully")

    async def test_workflow_runs_with_none_graph(self):
        """End-to-end: None graph_fragment -> failed status."""
        workflow = build_workflow()
        initial = _make_initial_state()
        result = await workflow.ainvoke(initial)
        if result["status"] != "failed":
            raise ValueError(f"Expected failed, got {result['status']}")

    async def test_workflow_runs_with_valid_graph(self):
        """End-to-end: Valid graph -> completed (Phase A: no detection)."""
        workflow = build_workflow()
        fragment = {
            "transactions": [{"id": "TX-1"}],
            "nodes": {"a": {"id": "a"}},
            "text_evidence": [],
            "ground_truth_criminals": [],
        }
        initial = _make_initial_state(graph_fragment=fragment)
        result = await workflow.ainvoke(initial)
        if result["status"] != "completed":
            raise ValueError(f"Expected completed, got {result['status']}")
        if result["detected_typology"] != "NONE":
            raise ValueError("Phase A should detect NONE (no heuristics wired)")
