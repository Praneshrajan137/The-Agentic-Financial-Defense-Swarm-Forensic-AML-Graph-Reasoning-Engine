"""
A2A Server -- FastAPI endpoint for Purple Agent
PRD Reference: Task A8

v11.0 FIXES (retained in v12.0):
- [HIGH-17] Exception handler restructured into separate except clauses.
v10.0 FIXES (retained):
- [CRIT-09] graph_fragment=None -> status="failed" is EXPECTED in Phase A.
- [HIGH-14] Lazy workflow initialization via @functools.lru_cache.
- [HIGH-15] Exception taxonomy: parse errors -> 400, bugs -> 500.
v9.0 FIXES (retained):
- [HIGH-10] hop_depth=0 validated (returns 422)
- [MED-17] Audit logging for every investigation
"""
import json
import logging
import time
import functools
from decimal import Decimal
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse

from src.config import (
    A2A_SERVER_HOST,
    A2A_SERVER_PORT,
    AGENT_VERSION,
    PROTOBUF_CONTENT_TYPE,
)
from src.core.decision_loop import build_workflow, InvestigationState
from protos import financial_crime_pb2 as pb2

logger = logging.getLogger(__name__)


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)


def json_response(data: dict[str, Any], status_code: int = 200) -> JSONResponse:
    content = json.loads(json.dumps(data, cls=DecimalEncoder))
    return JSONResponse(content=content, status_code=status_code)


app = FastAPI(
    title="Purple Agent -- Project Gamma",
    version=AGENT_VERSION,
    description="Autonomous forensic financial crime investigator",
)


@functools.lru_cache(maxsize=1)
def _get_workflow() -> Any:
    """Compile workflow on first use, cache thereafter."""
    return build_workflow()


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "purple_agent",
        "version": AGENT_VERSION,
    }


@app.get("/agent.json")
async def agent_card() -> JSONResponse:
    card_path = Path(__file__).parent.parent.parent / "agent.json"
    if not card_path.exists():
        raise HTTPException(status_code=404, detail="agent.json not found")
    with open(card_path) as f:
        card = json.load(f)
    return JSONResponse(content=card)


@app.post("/a2a")
async def handle_investigation(request: Request) -> Response:
    """Handle incoming investigation requests."""
    content_type = request.headers.get("content-type", "")
    body = await request.body()

    try:
        if PROTOBUF_CONTENT_TYPE in content_type:
            proto_req = pb2.InvestigationRequest()
            proto_req.ParseFromString(body)
            case_id = proto_req.case_id or f"CASE-{int(time.time())}"
            subject_id = proto_req.subject_id
            jurisdiction = proto_req.jurisdiction or "fincen"
            hop_depth = proto_req.hop_depth
        else:
            data = json.loads(body)
            case_id = data.get("case_id", f"CASE-{int(time.time())}")
            subject_id = data.get("subject_id") or data.get("graph_entry_point")
            jurisdiction = data.get("jurisdiction", "fincen")
            hop_depth = data.get("hop_depth", 0)

        if not subject_id:
            raise HTTPException(
                status_code=422,
                detail="subject_id (or graph_entry_point) is required",
            )

        if not isinstance(hop_depth, int) or hop_depth <= 0:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"hop_depth must be a positive integer, got {hop_depth}. "
                    f"Typical values: 2 (shallow), 3 (standard), 5 (deep)."
                ),
            )

    except json.JSONDecodeError as e:
        logger.warning("Malformed JSON in request: %s", e)
        raise HTTPException(status_code=400, detail=f"Malformed JSON: {e}")

    except HTTPException:
        raise

    except Exception as e:
        if (
            type(e).__name__ == "DecodeError"
            and hasattr(type(e), "__module__")
            and "protobuf" in getattr(type(e), "__module__", "")
        ):
            logger.warning("Malformed Protobuf in request: %s", e)
            raise HTTPException(
                status_code=400, detail=f"Malformed Protobuf: {e}"
            )
        raise

    initial_state: InvestigationState = {
        "case_id": case_id,
        "subject_id": subject_id,
        "jurisdiction": jurisdiction,
        "hop_depth": hop_depth,
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

    try:
        workflow = _get_workflow()
        result_state = await workflow.ainvoke(initial_state)
    except Exception as e:
        logger.error(
            "Pipeline failed for case %s: %s", case_id, e, exc_info=True
        )
        result_state = {
            **initial_state,
            "status": "failed",
            "error_message": str(e),
        }

    result_dict: dict[str, Any] = {
        "case_id": result_state["case_id"],
        "sar_narrative": result_state.get("sar_narrative", ""),
        "typology_detected": result_state.get("detected_typology", "NONE"),
        "involved_entities": sorted(
            result_state.get("involved_entities", [])
        ),
        "confidence_score": result_state.get("confidence_score", 0.0),
        "jurisdiction": result_state.get("jurisdiction", "fincen"),
        "investigation_timestamp": result_state.get(
            "investigation_timestamp", int(time.time())
        ),
        "status": result_state.get("status", "failed"),
        "error_message": result_state.get("error_message"),
    }

    logger.info(
        "Investigation complete: case_id=%s, subject_id=%s, status=%s, "
        "typology=%s, confidence=%.2f, entities=%d",
        result_dict["case_id"], subject_id, result_dict["status"],
        result_dict["typology_detected"],
        result_dict["confidence_score"],
        len(result_dict["involved_entities"]),
    )

    if PROTOBUF_CONTENT_TYPE in request.headers.get("accept", ""):
        proto_result = pb2.InvestigationResult(
            case_id=result_dict["case_id"],
            sar_narrative=result_dict["sar_narrative"] or "",
            typology_detected=result_dict["typology_detected"],
            confidence_score=float(result_dict["confidence_score"]),
            jurisdiction=result_dict["jurisdiction"],
            investigation_timestamp=result_dict["investigation_timestamp"],
        )
        proto_result.involved_entities.extend(result_dict["involved_entities"])
        return Response(
            content=proto_result.SerializeToString(),
            media_type=PROTOBUF_CONTENT_TYPE,
        )

    return json_response(result_dict)
