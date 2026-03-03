"""
LangGraph Decision Loop -- Investigation State Machine
PRD Reference: Task A7

v12.0 State transitions (8 nodes):
  ingest -> detect_structuring -> detect_layering -> synthesize_evidence
    -> compute_confidence --(HIGH)--> draft_sar -> validate_sar -> submit
                          \\--(LOW)---> submit (skip SAR, typology=NONE)
"""
import time
import logging
from typing import TypedDict, Any

from langgraph.graph import StateGraph, END

from src.config import SAR_MAX_RETRY, CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD_FLOAT: float = float(CONFIDENCE_THRESHOLD)


class InvestigationState(TypedDict):
    """Typed state dict for the investigation pipeline. 23 fields."""
    case_id: str
    subject_id: str
    jurisdiction: str
    hop_depth: int
    graph_fragment: dict | None
    transactions: list[dict]
    nodes: dict[str, dict]
    text_evidence: list[dict]
    ground_truth_criminals: list[str]
    structuring_results: list[dict]
    layering_results: list[dict]
    detected_typology: str | None
    involved_entities: list[str]
    evidence_results: list[dict]
    evidence_package: dict | None
    sar_narrative: str | None
    validation_result: dict | None
    confidence_score: float
    investigation_start_timestamp: int
    investigation_timestamp: int
    status: str
    retry_count: int
    error_message: str | None


def ingest(state: InvestigationState) -> InvestigationState:
    """Initialize investigation. Set status to in_progress."""
    now = int(time.time())
    if state.get("graph_fragment") is None:
        return {
            **state,
            "status": "failed",
            "error_message": (
                "Critical: graph_fragment is None. "
                "Cannot investigate empty graph (Rule 19)."
            ),
            "investigation_start_timestamp": now,
        }
    fragment = state["graph_fragment"]
    return {
        **state,
        "status": "in_progress",
        "transactions": fragment.get("transactions", []),
        "nodes": fragment.get("nodes", {}),
        "text_evidence": fragment.get("text_evidence", []),
        "ground_truth_criminals": fragment.get("ground_truth_criminals", []),
        "investigation_start_timestamp": now,
    }


def detect_structuring(state: InvestigationState) -> InvestigationState:
    """Run structuring (smurfing) detection heuristic.

    TODO(B2): Wire to src.core.heuristics.structuring.detect_structuring()
    """
    if state.get("status") == "failed":
        return state
    return {**state, "structuring_results": []}


def detect_layering(state: InvestigationState) -> InvestigationState:
    """Run layering detection heuristic.

    TODO(B3): Wire to src.core.heuristics.layering.detect_layering()
    """
    if state.get("status") == "failed":
        return state
    return {**state, "layering_results": []}


def synthesize_evidence(state: InvestigationState) -> InvestigationState:
    """Cross-reference text evidence with ledger data.

    TODO(B4): Wire to src.core.evidence_synthesizer.synthesize()
    """
    if state.get("status") == "failed":
        return state
    return {
        **state,
        "evidence_results": [],
        "evidence_package": {
            "text_corroborates_ledger": False,
            "has_suspicious_discrepancy": False,
            "discrepancies": [],
            "corroborations": [],
        },
    }


def compute_confidence(state: InvestigationState) -> InvestigationState:
    """Compute confidence_score and determine typology.

    Scoring Formula:
      base = 0.3 per detected typology, 0.6 if BOTH
      + 0.2 if text_corroborates_ledger
      + 0.2 if has_suspicious_discrepancy
      Clamped to [0.0, 1.0]. Gated on CONFIDENCE_THRESHOLD.
    """
    if state.get("status") == "failed":
        return state

    structuring = bool(state.get("structuring_results"))
    layering = bool(state.get("layering_results"))

    if structuring and layering:
        typology = "BOTH"
        base = 0.6
    elif structuring:
        typology = "STRUCTURING"
        base = 0.3
    elif layering:
        typology = "LAYERING"
        base = 0.3
    else:
        typology = "NONE"
        base = 0.0

    evidence_pkg = state.get("evidence_package") or {}
    evidence_boost = 0.2 if evidence_pkg.get("text_corroborates_ledger") else 0.0
    discrepancy_boost = (
        0.2 if evidence_pkg.get("has_suspicious_discrepancy") else 0.0
    )
    score = min(1.0, base + evidence_boost + discrepancy_boost)

    involved: set[str] = set()
    for r in state.get("structuring_results", []):
        involved.update(r.get("involved_entities", []))
    for r in state.get("layering_results", []):
        involved.update(r.get("involved_entities", []))

    if score < _CONFIDENCE_THRESHOLD_FLOAT:
        typology = "NONE"
        involved = set()

    return {
        **state,
        "detected_typology": typology,
        "confidence_score": score,
        "involved_entities": sorted(involved),
    }


def should_generate_sar(state: InvestigationState) -> str:
    """Conditional edge after compute_confidence."""
    if state.get("status") == "failed":
        return "submit"
    if state.get("detected_typology", "NONE") == "NONE":
        return "submit"
    return "draft"


def draft_sar(state: InvestigationState) -> InvestigationState:
    """Generate SAR narrative.

    TODO(C1): Wire to src.core.sar_drafter.generate_sar()
    """
    if state.get("status") == "failed":
        return state
    if state.get("sar_narrative") is None:
        state = {**state, "sar_narrative": "Draft SAR placeholder"}
    return state


def validate_sar(state: InvestigationState) -> InvestigationState:
    """Validate SAR narrative against graph data (structural checks).

    Phase A performs basic structural validation:
      1. sar_narrative must be non-None and non-empty.
      2. If involved_entities exist, at least one must appear in the narrative.

    v9.0 [MED-11 FIX]: On validation FAILURE, clears sar_narrative to None
    so that draft_sar will re-generate on retry.

    TODO(C2): Wire full validation (every cited entity/amount/timestamp
    must exist in the graph).
    """
    if state.get("status") == "failed":
        return {
            **state,
            "validation_result": {
                "passed": False, "errors": ["Investigation failed"]
            },
        }

    new_retry = state.get("retry_count", 0) + 1
    errors: list[str] = []

    narrative = state.get("sar_narrative")
    if not narrative or not narrative.strip():
        errors.append("SAR narrative is empty or missing")

    entities = state.get("involved_entities") or []
    if narrative and entities:
        referenced = [e for e in entities if e in narrative]
        if not referenced:
            errors.append(
                f"No involved entities referenced in narrative "
                f"(expected at least one of {sorted(entities)[:5]})"
            )

    if errors:
        logger.warning("SAR validation failed (retry %d): %s", new_retry, errors)
        return {
            **state,
            "validation_result": {"passed": False, "errors": errors},
            "retry_count": new_retry,
            "sar_narrative": None,
        }
    return {
        **state,
        "validation_result": {"passed": True, "errors": []},
        "retry_count": new_retry,
    }


def should_retry(state: InvestigationState) -> str:
    """Conditional edge: retry SAR drafting if validation fails."""
    validation = state.get("validation_result") or {}
    if validation.get("passed", False):
        return "submit"
    if state.get("retry_count", 0) >= SAR_MAX_RETRY:
        return "submit"
    return "draft"


def submit(state: InvestigationState) -> InvestigationState:
    """Mark investigation complete."""
    final_status = "completed" if state.get("status") != "failed" else "failed"
    return {
        **state,
        "status": final_status,
        "investigation_timestamp": int(time.time()),
    }


def build_workflow() -> Any:
    """Build and compile the LangGraph investigation workflow (8 nodes)."""
    workflow = StateGraph(InvestigationState)

    workflow.add_node("ingest", ingest)
    workflow.add_node("detect_structuring", detect_structuring)
    workflow.add_node("detect_layering", detect_layering)
    workflow.add_node("synthesize_evidence", synthesize_evidence)
    workflow.add_node("compute_confidence", compute_confidence)
    workflow.add_node("draft_sar", draft_sar)
    workflow.add_node("validate_sar", validate_sar)
    workflow.add_node("submit", submit)

    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "detect_structuring")
    workflow.add_edge("detect_structuring", "detect_layering")
    workflow.add_edge("detect_layering", "synthesize_evidence")
    workflow.add_edge("synthesize_evidence", "compute_confidence")
    workflow.add_conditional_edges(
        "compute_confidence", should_generate_sar,
        {"draft": "draft_sar", "submit": "submit"},
    )
    workflow.add_edge("draft_sar", "validate_sar")
    workflow.add_conditional_edges(
        "validate_sar", should_retry,
        {"submit": "submit", "draft": "draft_sar"},
    )
    workflow.add_edge("submit", END)

    return workflow.compile()
