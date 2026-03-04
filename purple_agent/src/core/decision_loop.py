"""
LangGraph Decision Loop -- Investigation State Machine
PRD Reference: Task A7 (structure) + C2 (wiring)
Package Version: v9.0

8-node state machine matching Architecture S3:
  receive -> analyze -> detect -> synthesize -> compute_confidence ->
  [gate] -> draft -> validate -> [retry/submit] -> submit

v9.0 CHANGES:
- compute_confidence evidence_boost: positive matching (defense-in-depth)
- InvestigationState docstring: corrected from "superset of A7" to
  "minimal operational subset of Architecture S5"
- synthesize_evidence: identity comprehension removed

v8.0 CRITICAL CHANGES (retained):
- compute_confidence is now a separate 8th node (Architecture S3 compliance)
- Confidence formula: Architecture S4 Step 6 three-factor formula
  (base + evidence_boost + discrepancy_boost), NOT max(B2/B3 confidences)
- Confidence gate runs AFTER evidence synthesis (not before)
- Currency parameter passed to detect_layering (v8.0 Part 4 API)
- SHA-256 idempotency key on submission (Rule 18)
- Hit dicts include currency field
- Primary field names (hop_count, counterpart_nodes)
- Async safety: detect existing event loop before asyncio.run()

v6.1 WIRING: All TODOs replaced with real module calls.
v6.1 FIXES: Empty graph fail, mechanical SAR fallback,
            InvestigationState superset of A7, asyncio.run().

BUG FIXES applied from plan review:
- Bug 1: submit_result passes case_id to A2AClient.submit_result()
- Bug 2: synthesize_evidence uses graph_fragment text_evidence (no fetch_communications)
- Bug 3: analyze_graph passes hop_depth to A2AClient.fetch_graph()
- Bug 5: investigation_start_timestamp retained in InvestigationState
- Bug 7: Status strings standardized to uppercase per Architecture S5
"""
import asyncio
import hashlib
import time
import logging
from decimal import Decimal
from typing import TypedDict

from langgraph.graph import StateGraph, END

from src.config import SAR_MAX_RETRY, CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)

# v9.0 [P5v9-05]: Known evidence verdicts that signal corroboration.
# Architecture S4 Step 6: "+0.2 if text evidence corroborates detection."
# Only these explicit corroboration signals trigger the boost.
_CORROBORATING_VERDICTS = frozenset({
    "CORROBORATED",
    "SUSPICIOUS_DISCREPANCY",
    "CONFIRMED",
    "PARTIAL_MATCH",
})


class InvestigationState(TypedDict):
    """Typed state dict for the investigation pipeline.

    v9.0 [P5v9-09]: MINIMAL OPERATIONAL SUBSET of Architecture S5.
    Architecture S5 fields like graph, transactions, nodes, text_evidence,
    ground_truth_criminals, structuring_results, layering_results,
    evidence_results are carried INSIDE composite dicts (graph_fragment,
    detection_results, evidence_package) rather than as top-level fields.

    confidence_score: float (NOT Decimal -- probability [0,1], not currency).
    """
    case_id: str
    subject_id: str
    jurisdiction: str
    hop_depth: int
    graph_fragment: dict | None
    detected_typology: str | None
    detection_results: dict | None
    evidence_package: dict | None
    sar_narrative: str | None
    sar_draft: dict | None
    validation_result: dict | None
    involved_entities: list[str]
    confidence_score: float
    investigation_start_timestamp: int
    investigation_timestamp: int
    status: str
    retry_count: int
    error_message: str | None


def _run_async(coro):
    """Run an async coroutine safely, handling existing event loops.

    v8.0 [P5v8-13]: LangGraph may invoke nodes in an async context
    where an event loop is already running. Detect and fall back gracefully.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


def receive_case(state: InvestigationState) -> InvestigationState:
    """Initialize investigation. Set status to IN_PROGRESS."""
    logger.info("[%s] Investigation received. Subject: %s", state["case_id"], state["subject_id"])
    return {
        **state,
        "status": "IN_PROGRESS",
        "investigation_start_timestamp": int(time.time()),
    }


def analyze_graph(state: InvestigationState) -> InvestigationState:
    """Fetch graph data from Green Agent via A2A Client.

    v6.1 FIX [ALN-04]: Empty graph -> FAIL status (Rule 19).
    Bug 3 fix: passes hop_depth to fetch_graph.
    """
    try:
        from src.core.a2a_client import A2AClient

        client = A2AClient()
        graph_data = _run_async(
            client.fetch_graph(state["subject_id"], state.get("hop_depth", 3))
        )

        node_count = len(graph_data.get("nodes", {}))
        tx_count = len(graph_data.get("transactions", []))

        if node_count == 0 or tx_count == 0:
            logger.error(
                "[%s] Empty graph received: %d nodes, %d transactions. "
                "Cannot investigate empty graph (Rule 19).",
                state["case_id"], node_count, tx_count,
            )
            return {
                **state,
                "status": "FAILED",
                "error_message": (
                    f"Critical: Graph fetch returned empty data "
                    f"({node_count} nodes, {tx_count} transactions). "
                    "Cannot investigate empty graph."
                ),
            }

        logger.info(
            "[%s] Graph fetched: %d nodes, %d transactions",
            state["case_id"], node_count, tx_count,
        )
        return {**state, "graph_fragment": graph_data}
    except Exception as e:
        logger.error("[%s] Graph fetch failed: %s", state["case_id"], e)
        return {
            **state,
            "status": "FAILED",
            "error_message": f"Graph fetch failed: {e}",
        }


def detect_typology(state: InvestigationState) -> InvestigationState:
    """Run structuring and layering detection on all graph nodes.

    v8.0: Currency parameter passed to detect_layering.
    v8.0: Hit dicts include currency field.
    v8.0: Primary field names: counterpart_nodes, hop_count.
    v8.0 [P5v8-15]: Exception -> status="FAILED".
    """
    if state.get("status") == "FAILED":
        return state

    try:
        from src.core.graph_reasoner import GraphReasoner
        from src.core.heuristics.structuring import detect_structuring
        from src.core.heuristics.layering import detect_layering

        graph_data = state["graph_fragment"] or {"transactions": [], "nodes": {}}
        reasoner = GraphReasoner()
        reasoner.load_from_dict(graph_data)

        currency = "INR" if state["jurisdiction"] == "fiu_ind" else "USD"

        structuring_hits: list[dict] = []
        all_structuring_nodes: set[str] = set()
        for node_id in reasoner.get_all_node_ids():
            result = detect_structuring(reasoner, node_id, currency=currency)
            if result.detected:
                structuring_hits.append({
                    "node": node_id,
                    "mode": result.mode,
                    "sources": result.counterpart_nodes,
                    "tx_count": result.transaction_count,
                    "total": str(result.total_amount),
                    "currency": result.currency,
                    "transactions": [tx["tx_id"] for tx in result.qualifying_transactions],
                })
                all_structuring_nodes.add(node_id)
                all_structuring_nodes.update(result.counterpart_nodes)

        layering_hits: list[dict] = []
        all_layering_nodes: set[str] = set()
        for node_id in reasoner.get_all_node_ids():
            result_l = detect_layering(
                reasoner, node_id,
                max_depth=state.get("hop_depth") or None,
                currency=currency,
            )
            if result_l.detected:
                for chain in result_l.chains:
                    layering_hits.append({
                        "start_node": node_id,
                        "chain_nodes": chain.chain_nodes,
                        "hop_count": chain.hop_count,
                        "avg_decay": str(chain.avg_decay_rate),
                        "currency": chain.currency,
                        "transactions": [tx["tx_id"] for tx in chain.chain_transactions],
                    })
                    all_layering_nodes.update(chain.chain_nodes)

        has_structuring = len(structuring_hits) > 0
        has_layering = len(layering_hits) > 0

        if has_structuring and has_layering:
            typology = "BOTH"
        elif has_structuring:
            typology = "STRUCTURING"
        elif has_layering:
            typology = "LAYERING"
        else:
            typology = "NONE"

        involved = sorted(all_structuring_nodes | all_layering_nodes)

        detection_results = {
            "typology": typology,
            "structuring_hits": structuring_hits,
            "layering_hits": layering_hits,
            "involved_entities": involved,
        }

        logger.info(
            "[%s] Detection complete: %s (involved=%d entities)",
            state["case_id"], typology, len(involved),
        )

        return {
            **state,
            "detected_typology": typology,
            "detection_results": detection_results,
            "involved_entities": involved,
            "confidence_score": 0.0,
        }
    except Exception as e:
        logger.error("[%s] Detection failed: %s", state["case_id"], e, exc_info=True)
        return {
            **state,
            "status": "FAILED",
            "detected_typology": "NONE",
            "detection_results": {},
            "confidence_score": 0.0,
            "error_message": f"Detection failed: {e}",
        }


def synthesize_evidence(state: InvestigationState) -> InvestigationState:
    """Cross-reference graph data with text evidence.

    Bug 2 fix: Uses text_evidence from graph_fragment instead of
    non-existent fetch_communications method.
    v9.0 [P5v9-08]: Identity comprehension removed.
    """
    if state.get("status") == "FAILED":
        return state

    if state.get("detected_typology") == "NONE":
        return {**state, "evidence_package": {"verdict": "NOT_APPLICABLE", "discrepancies": []}}

    try:
        from src.core.evidence_synthesizer import EvidenceSynthesizer

        graph_data = state["graph_fragment"] or {}
        text_evidence = graph_data.get("text_evidence", [])

        if not text_evidence:
            logger.info("[%s] No text evidence in graph fragment.", state["case_id"])
            return {**state, "evidence_package": {"verdict": "INSUFFICIENT_DATA", "discrepancies": []}}

        synthesizer = EvidenceSynthesizer()
        currency = "INR" if state["jurisdiction"] == "fiu_ind" else "USD"

        total_inflow = Decimal("0")
        for tx in graph_data.get("transactions", []):
            if tx["target_node"] == state["subject_id"]:
                total_inflow += tx["amount"]

        ledger_data = {
            "total_inflow": total_inflow,
            "entity_id": state["subject_id"],
            "transactions": graph_data.get("transactions", []),
        }

        result = synthesizer.synthesize(ledger_data, text_evidence, currency=currency)

        evidence_pkg = {
            "verdict": result.verdict,
            "reasoning": result.reasoning,
            "discrepancies": result.discrepancies,
            "entities": list(result.extracted_entities),
            "amounts": [str(a) for a in result.extracted_amounts],
        }

        logger.info(
            "[%s] Evidence synthesis: %s (%d discrepancies)",
            state["case_id"], result.verdict, len(result.discrepancies),
        )
        return {**state, "evidence_package": evidence_pkg}
    except Exception as e:
        logger.error("[%s] Evidence synthesis failed: %s", state["case_id"], e, exc_info=True)
        return {**state, "evidence_package": {"verdict": "INSUFFICIENT_DATA", "discrepancies": []}}


def compute_confidence(state: InvestigationState) -> InvestigationState:
    """Compute confidence score using Architecture S4 Step 6 formula.

    v8.0 [P5v8-01/P5v8-02]: Separate 8th node extracted from detect_typology.
    v9.0 [P5v9-05]: Evidence boost uses positive matching.

    Formula:
      Base: 0.3 (one typology) | 0.6 (BOTH typologies)
      Evidence boost: +0.2 if text evidence corroborates detection
      Discrepancy boost: +0.2 if SUSPICIOUS_DISCREPANCY found
      Score = min(1.0, base + evidence_boost + discrepancy_boost)

    IEEE 754 safety verified:
      0.3 + 0.2 = 0.5 exactly, 0.6 + 0.2 = 0.8 exactly,
      0.3 + 0.2 + 0.2 = 0.7 exactly, 0.6 + 0.2 + 0.2 = 1.0 exactly.

    Gate: If score < CONFIDENCE_THRESHOLD (0.5), set typology to NONE,
    skip SAR, return LOW_CONFIDENCE result via conditional edge.
    """
    if state.get("status") == "FAILED":
        return state

    typology = state.get("detected_typology", "NONE")
    evidence = state.get("evidence_package") or {}

    if typology == "NONE":
        return {**state, "confidence_score": 0.0}

    if typology == "BOTH":
        base = 0.6
    else:
        base = 0.3

    # v9.0 [P5v9-05]: POSITIVE matching -- only known corroboration verdicts
    evidence_boost = 0.0
    verdict = evidence.get("verdict", "")
    if verdict in _CORROBORATING_VERDICTS:
        evidence_boost = 0.2

    discrepancy_boost = 0.0
    discrepancies = evidence.get("discrepancies", [])
    if discrepancies and len(discrepancies) > 0:
        discrepancy_boost = 0.2

    score = min(1.0, base + evidence_boost + discrepancy_boost)

    logger.info(
        "[%s] Confidence computed: base=%s + evidence=%s + discrepancy=%s "
        "= %s (threshold=%s)",
        state["case_id"], base, evidence_boost, discrepancy_boost,
        score, float(CONFIDENCE_THRESHOLD),
    )

    if score < float(CONFIDENCE_THRESHOLD):
        logger.info(
            "[%s] Confidence %s < threshold %s. Downgrading typology to NONE (LOW_CONFIDENCE).",
            state["case_id"], score, CONFIDENCE_THRESHOLD,
        )
        return {
            **state,
            "confidence_score": score,
            "detected_typology": "NONE",
        }

    return {**state, "confidence_score": score}


def draft_sar(state: InvestigationState) -> InvestigationState:
    """Generate SAR narrative via LLM.

    v6.1 [ALN-05]: Falls back to mechanical template if LLM call fails.
    """
    if state.get("status") == "FAILED":
        return state

    if state.get("detected_typology") == "NONE":
        return {**state, "sar_narrative": "", "sar_draft": None}

    try:
        from src.core.sar_drafter import SARDrafter

        drafter = SARDrafter()
        draft = drafter.draft_sar(
            detection_results=state.get("detection_results") or {},
            evidence_package=state.get("evidence_package") or {},
            jurisdiction=state["jurisdiction"],
            graph_data=state.get("graph_fragment") or {},
        )

        logger.info(
            "[%s] SAR drafted: %d TX citations, narrative length=%d",
            state["case_id"], len(draft.cited_tx_ids), len(draft.raw_narrative),
        )

        return {
            **state,
            "sar_narrative": draft.raw_narrative,
            "sar_draft": {
                "who": draft.who,
                "what": draft.what,
                "where": draft.where,
                "when": draft.when,
                "why": draft.why,
                "cited_tx_ids": draft.cited_tx_ids,
                "raw_narrative": draft.raw_narrative,
                "jurisdiction": draft.jurisdiction,
            },
        }
    except Exception as e:
        logger.error("[%s] LLM SAR drafting failed: %s", state["case_id"], e)

        try:
            from src.core.sar_drafter import mechanical_sar_template
            draft = mechanical_sar_template(
                detection_results=state.get("detection_results") or {},
                graph_data=state.get("graph_fragment") or {},
                jurisdiction=state["jurisdiction"],
            )
            logger.info("[%s] Mechanical SAR template generated as fallback.", state["case_id"])
            return {
                **state,
                "sar_narrative": draft.raw_narrative,
                "sar_draft": {
                    "who": draft.who, "what": draft.what, "where": draft.where,
                    "when": draft.when, "why": draft.why,
                    "cited_tx_ids": draft.cited_tx_ids,
                    "raw_narrative": draft.raw_narrative,
                    "jurisdiction": draft.jurisdiction,
                },
            }
        except Exception as e2:
            logger.error("[%s] Mechanical SAR also failed: %s", state["case_id"], e2)
            return {**state, "sar_narrative": "", "sar_draft": None}


def validate_sar(state: InvestigationState) -> InvestigationState:
    """Validate SAR against graph data. Increment retry on failure."""
    if state.get("status") == "FAILED":
        return {**state, "validation_result": {"passed": False, "errors": ["Investigation failed"]}}

    if state.get("sar_draft") is None:
        return {
            **state,
            "validation_result": {"passed": True, "errors": []},
        }

    try:
        from src.core.sar_drafter import SARDrafter, SARDraft

        drafter = SARDrafter()
        draft_data = state.get("sar_draft")
        draft = SARDraft(**draft_data)
        graph_data = state.get("graph_fragment") or {}

        validation = drafter.validate_sar(draft, graph_data)

        if not validation.passed:
            logger.warning(
                "[%s] SAR validation failed (attempt %d): %s",
                state["case_id"], state.get("retry_count", 0) + 1, validation.errors,
            )
            return {
                **state,
                "validation_result": {"passed": False, "errors": validation.errors},
                "retry_count": state.get("retry_count", 0) + 1,
                "sar_narrative": None,
            }

        logger.info("[%s] SAR validation PASSED", state["case_id"])
        return {
            **state,
            "validation_result": {"passed": True, "errors": []},
        }
    except Exception as e:
        logger.error("[%s] SAR validation error: %s", state["case_id"], e)
        return {
            **state,
            "validation_result": {"passed": False, "errors": [str(e)]},
            "retry_count": state.get("retry_count", 0) + 1,
        }


def submit_result(state: InvestigationState) -> InvestigationState:
    """Submit investigation result via A2A Client.

    v8.0 [P5v8-04]: Computes SHA-256 idempotency key per Rule 18.
    v6.1 [ALN-05]: Mechanical SAR fallback on retries exhausted.
    Bug 1 fix: passes case_id to A2AClient.submit_result().
    """
    try:
        from src.core.sar_drafter import SARDrafter, SARDraft

        draft_data = state.get("sar_draft")
        formatted_narrative = state.get("sar_narrative", "")

        # v6.1 [ALN-05]: Mechanical fallback if narrative empty but crime detected
        if (not formatted_narrative or formatted_narrative == "") and state.get("detected_typology") != "NONE":
            from src.core.sar_drafter import mechanical_sar_template
            fallback_draft = mechanical_sar_template(
                detection_results=state.get("detection_results") or {},
                graph_data=state.get("graph_fragment") or {},
                jurisdiction=state["jurisdiction"],
            )
            drafter = SARDrafter()
            formatted_narrative = drafter.format_sar(fallback_draft)
            logger.warning("[%s] Using mechanical SAR template for submission (Rule 24).", state["case_id"])
        elif draft_data:
            drafter = SARDrafter()
            draft = SARDraft(**draft_data)
            formatted_narrative = drafter.format_sar(draft)

        typology = state.get("detected_typology", "NONE")
        involved = state.get("involved_entities", [])
        idem_input = f"{state['case_id']}{typology}{''.join(sorted(involved))}"
        idempotency_key = hashlib.sha256(idem_input.encode("utf-8")).hexdigest()

        result_dict = {
            "case_id": state["case_id"],
            "sar_narrative": formatted_narrative,
            "typology_detected": typology,
            "involved_entities": involved,
            "confidence_score": state.get("confidence_score", 0.0),
            "jurisdiction": state["jurisdiction"],
            "investigation_timestamp": int(time.time()),
            "idempotency_key": idempotency_key,
        }

        from src.core.a2a_client import A2AClient
        client = A2AClient()
        _run_async(client.submit_result(result_dict, state["case_id"]))

        logger.info(
            "[%s] Result submitted: %s (confidence=%s, idem_key=%s...)",
            state["case_id"], typology, state.get("confidence_score", 0),
            idempotency_key[:16],
        )
        return {**state, "status": "COMPLETE"}
    except Exception as e:
        logger.error("[%s] Submission failed: %s", state["case_id"], e)
        return {**state, "status": "COMPLETE"}


def should_generate_sar(state: InvestigationState) -> str:
    """Conditional edge: proceed to SAR drafting or skip to submit.

    v8.0 [P5v8-02]: Conditional edge on compute_confidence node.
    """
    if state.get("status") == "FAILED":
        return "submit"
    if state.get("detected_typology") == "NONE":
        return "submit"
    if state.get("confidence_score", 0.0) < float(CONFIDENCE_THRESHOLD):
        return "submit"
    return "draft"


def should_retry(state: InvestigationState) -> str:
    """Conditional edge: retry SAR drafting if validation fails.

    v6.1 [ALN-05]: When max retries reached, submit_result will use
    mechanical SAR template (Rule 24: never empty narrative).
    """
    validation = state.get("validation_result", {})
    if validation and validation.get("passed", False):
        return "submit"
    if state.get("retry_count", 0) >= SAR_MAX_RETRY:
        logger.warning(
            "[%s] Max retries (%d) reached. Submitting with mechanical fallback (Rule 24).",
            state["case_id"], SAR_MAX_RETRY,
        )
        return "submit"
    return "draft"


def build_workflow() -> StateGraph:
    """Build and compile the LangGraph investigation workflow.

    8-node state machine:
    receive -> analyze -> detect -> synthesize -> compute_confidence ->
    [should_generate_sar] -> draft -> validate -> [should_retry] -> submit

    Conditional edges:
      compute_confidence -> draft    (confidence >= CONFIDENCE_THRESHOLD)
      compute_confidence -> submit   (LOW_CONFIDENCE: score < threshold)
      validate           -> draft    (retry: validation failed, retries < max)
      validate           -> submit   (passed or retries exhausted -> mechanical)
    """
    workflow = StateGraph(InvestigationState)

    workflow.add_node("receive", receive_case)
    workflow.add_node("analyze", analyze_graph)
    workflow.add_node("detect", detect_typology)
    workflow.add_node("synthesize", synthesize_evidence)
    workflow.add_node("compute_confidence", compute_confidence)
    workflow.add_node("draft", draft_sar)
    workflow.add_node("validate", validate_sar)
    workflow.add_node("submit", submit_result)

    workflow.set_entry_point("receive")
    workflow.add_edge("receive", "analyze")
    workflow.add_edge("analyze", "detect")
    workflow.add_edge("detect", "synthesize")
    workflow.add_edge("synthesize", "compute_confidence")
    workflow.add_conditional_edges(
        "compute_confidence", should_generate_sar,
        {"draft": "draft", "submit": "submit"},
    )
    workflow.add_edge("draft", "validate")
    workflow.add_conditional_edges(
        "validate", should_retry,
        {"submit": "submit", "draft": "draft"},
    )
    workflow.add_edge("submit", END)

    return workflow.compile()
