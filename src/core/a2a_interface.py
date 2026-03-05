"""
Agent2Agent Interface Module
============================
Exposes synthetic financial crime data via A2A protocol.

Protocol Specification:
- POST /a2a -- A2A protocol endpoint (InvestigationRequest -> GraphFragment)
- POST /results -- Investigation result submission
- POST /a2a/tools/* -- Incremental data exposure tools
- GET /health -- Health check
- GET /agent.json -- Agent manifest
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response as FastAPIResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from collections import deque
from enum import Enum
import networkx as nx
import uuid
import time
import logging

from src.config import (
    AGENT_VERSION,
    A2A_SERVER_PORT,
    PROTOBUF_CONTENT_TYPE,
    DEFAULT_HOP_DEPTH,
    MAX_FRAGMENT_TRANSACTIONS,
    MAX_FRAGMENT_NODES,
)

logger = logging.getLogger(__name__)

# Import protobuf module with fallback
try:
    from . import financial_crime_pb2 as pb2
    PROTOBUF_AVAILABLE = True
except ImportError:
    PROTOBUF_AVAILABLE = False
    pb2 = None  # type: ignore[assignment]


# Enums
class EntityType(str, Enum):
    PERSON = "person"
    COMPANY = "company"
    BANK = "bank"


class TransactionType(str, Enum):
    WIRE = "wire"
    ACH = "ach"
    CASH = "cash"
    INTERNAL = "internal"


class TransactionDirection(str, Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    BOTH = "both"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    PENDING = "pending"
    FAILED = "failed"


class ConnectionRelationship(str, Enum):
    SENDER = "sender"
    RECEIVER = "receiver"
    BOTH = "both"


# FastAPI application
app = FastAPI(
    title="Green Financial Crime Agent",
    description="A2A Protocol interface for synthetic AML data",
    version=AGENT_VERSION,
)


# Middleware for tracking tool calls
@app.middleware("http")
async def track_tool_calls(request: Request, call_next):
    """
    Track tool usage for efficiency scoring.
    
    Every call to /a2a/tools/* increments the counter for the participant.
    Participant ID is extracted from X-Participant-ID header.
    """
    # Extract participant_id from header
    participant_id = request.headers.get("X-Participant-ID")
    
    # Call the endpoint
    response = await call_next(request)
    
    # Increment counter if this was a tool call and we have a participant
    if participant_id and request.url.path.startswith('/a2a/tools/'):
        count = increment_tool_counter(participant_id)
        # Add custom header showing current count
        response.headers['X-Tool-Call-Count'] = str(count)
    
    return response


# Request/Response Models
class AgentManifest(BaseModel):
    """Agent manifest for A2A protocol discovery."""
    name: str = "green-financial-crime-agent"
    version: str = AGENT_VERSION
    description: str = "Synthetic financial crime data generator with evidence artifacts"
    capabilities: List[str] = [
        "generate_graph",
        "inject_structuring",
        "inject_layering",
        "get_transactions",
        "get_kyc_profile",
        "get_evidence",
        "a2a_graph_fragment",
        "investigation_results",
    ]
    endpoints: Dict[str, str] = {
        "health": "/health",
        "manifest": "/agent.json",
        "a2a": "/a2a",
        "results": "/results",
        "transactions": "/a2a/tools/get_transactions",
        "kyc": "/a2a/tools/get_kyc_profile",
        "connections": "/a2a/tools/get_account_connections",
        "evidence": "/a2a/tools/get_evidence",
        "assessment": "/a2a/investigation_assessment",
    }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = AGENT_VERSION
    graph_loaded: bool = False
    node_count: int = 0
    edge_count: int = 0


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


# Investigation Assessment
class InvestigationAssessmentRequest(BaseModel):
    """Request for investigation assessment."""
    participant_id: str
    investigation_data: Optional[Dict[str, Any]] = None


class RubricBreakdown(BaseModel):
    """Detailed scoring breakdown."""
    pattern_identification: float = 0.0
    evidence_quality: float = 0.0
    narrative_clarity: float = 0.0
    completeness: float = 0.0


class InvestigationAssessmentResponse(BaseModel):
    """Response for investigation assessment with efficiency metrics."""
    score: float = Field(ge=0, le=100)
    feedback: str
    rubric_breakdown: Optional[RubricBreakdown] = None
    missed_indicators: Optional[List[str]] = None
    tool_call_count: int = 0  # Number of tool calls made
    efficiency_score: float = 0.0  # Efficiency score (0-100)
    efficiency_rank: str = "unknown"  # Rank: "excellent", "good", "fair", "poor"


# Transaction Endpoints
class GetTransactionsRequest(BaseModel):
    """Request for transaction history."""
    account_id: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(default=100, ge=1, le=1000)
    direction: TransactionDirection = TransactionDirection.BOTH


class Transaction(BaseModel):
    """Transaction record."""
    transaction_id: str
    source: str
    target: str
    amount: float
    currency: str = "USD"
    timestamp: datetime
    transaction_type: TransactionType = TransactionType.WIRE
    memo: Optional[str] = None


class TransactionListResponse(BaseModel):
    """Response containing transaction list."""
    account_id: str
    transaction_count: int
    transactions: List[Transaction]


# KYC Endpoints
class GetKycProfileRequest(BaseModel):
    """Request for KYC profile."""
    account_id: str


class KycProfileResponse(BaseModel):
    """KYC profile response."""
    account_id: str
    entity_type: EntityType
    name: str
    company: Optional[str] = None
    address: Optional[str] = None
    country: Optional[str] = None
    swift_code: Optional[str] = None
    iban: Optional[str] = None
    risk_score: float = Field(ge=0, le=1)
    verification_status: VerificationStatus = VerificationStatus.VERIFIED
    created_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None


# Connection Endpoints
class GetAccountConnectionsRequest(BaseModel):
    """Request for account connections."""
    account_id: str
    depth: int = Field(default=1, ge=1, le=3)


class AccountConnection(BaseModel):
    """Single account connection."""
    account_id: str
    relationship: ConnectionRelationship
    transaction_count: int
    total_amount: float


class AccountConnectionsResponse(BaseModel):
    """Response containing account connections."""
    account_id: str
    connection_count: int
    connections: List[AccountConnection]


# Evidence Endpoints
class GetEvidenceRequest(BaseModel):
    """Request for evidence documents."""
    entity_id: Optional[str] = None  # Filter by entity
    document_type: Optional[str] = None  # Filter by type (SAR, email, etc.)
    contains_keyword: Optional[str] = None  # Search in text
    limit: int = Field(default=100, ge=1, le=1000)


class EvidenceDocument(BaseModel):
    """Evidence document structure."""
    document_id: str
    document_type: str
    subject_id: Optional[str] = None
    date: datetime
    content: str
    metadata: Dict[str, Any] = {}


class EvidenceListResponse(BaseModel):
    """Response containing evidence documents."""
    document_count: int
    documents: List[EvidenceDocument]


# Global state (will be managed by persistent state in production)
_current_graph: Optional[Union[nx.DiGraph, nx.MultiDiGraph]] = None
_injected_crimes: List[Dict[str, Any]] = []
_ground_truth: Dict[str, Any] = {}
_evidence_documents: List[Dict[str, Any]] = []

# Investigation results storage (keyed by idempotency key)
_investigation_results: Dict[str, Dict[str, Any]] = {}

# Tool call tracking for efficiency metrics
_tool_call_counters: Dict[str, int] = {}


def set_graph(graph: Union[nx.DiGraph, nx.MultiDiGraph]) -> None:
    """Set the current graph for the API."""
    global _current_graph
    _current_graph = graph


def set_ground_truth(truth: Dict[str, Any]) -> None:
    """Set the ground truth for assessment."""
    global _ground_truth
    _ground_truth = truth


def set_evidence(evidence_list: List[Dict[str, Any]]) -> None:
    """Set evidence documents for the API."""
    global _evidence_documents
    _evidence_documents = evidence_list


def reset_tool_counter(participant_id: str) -> None:
    """Reset tool call counter for a participant."""
    global _tool_call_counters
    _tool_call_counters[participant_id] = 0


def increment_tool_counter(participant_id: str) -> int:
    """
    Increment and return tool call count for a participant.
    
    Args:
        participant_id: Unique identifier for the participant
        
    Returns:
        Updated tool call count
    """
    global _tool_call_counters
    if participant_id not in _tool_call_counters:
        _tool_call_counters[participant_id] = 0
    _tool_call_counters[participant_id] += 1
    return _tool_call_counters[participant_id]


def get_tool_count(participant_id: str) -> int:
    """
    Get current tool call count for a participant.
    
    Args:
        participant_id: Unique identifier for the participant
        
    Returns:
        Current tool call count (0 if participant not found)
    """
    return _tool_call_counters.get(participant_id, 0)


def _resolve_account_id(account_id: str) -> Any:
    """
    Resolve account_id to the appropriate type for graph lookup.
    
    Graph nodes may be integers or strings; this tries to match either.
    
    Args:
        account_id: String account ID from request
        
    Returns:
        The resolved account ID (may be int or str)
    """
    if _current_graph is None:
        return account_id
    
    # First check if string version exists
    if account_id in _current_graph.nodes():
        return account_id
    
    # Try as integer
    try:
        int_id = int(account_id)
        if int_id in _current_graph.nodes():
            return int_id
    except ValueError:
        pass
    
    # Return original if no match found
    return account_id


# System Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=AGENT_VERSION,
        graph_loaded=_current_graph is not None,
        node_count=_current_graph.number_of_nodes() if _current_graph else 0,
        edge_count=_current_graph.number_of_edges() if _current_graph else 0
    )


@app.get("/agent.json", response_model=AgentManifest)
async def get_manifest() -> AgentManifest:
    """Return agent manifest for A2A discovery."""
    return AgentManifest()


# =============================================================================
# A2A Protocol Endpoints (Purple Agent contract)
# =============================================================================

def _datetime_to_epoch(dt: Any) -> int:
    """Convert datetime to Unix epoch seconds (int64).

    Handles datetime objects, ISO strings, and passthrough for ints.
    Returns 0 for unconvertible values (defensive, logged upstream).
    """
    if isinstance(dt, int):
        return dt
    if isinstance(dt, float):
        return int(dt)
    if isinstance(dt, datetime):
        return int(dt.timestamp())
    if isinstance(dt, str):
        try:
            return int(datetime.fromisoformat(dt).timestamp())
        except (ValueError, TypeError):
            return 0
    return 0


def _bfs_subgraph(
    graph: nx.DiGraph,
    subject_id: Any,
    hop_depth: int,
) -> set:
    """BFS traversal from subject_id up to hop_depth hops.

    Returns the set of node IDs reachable within hop_depth.
    Operates on both DiGraph and MultiDiGraph.
    """
    visited: set = set()
    queue: deque = deque()
    queue.append((subject_id, 0))
    visited.add(subject_id)

    while queue:
        node, depth = queue.popleft()
        if depth >= hop_depth:
            continue
        for neighbor in set(graph.successors(node)) | set(graph.predecessors(node)):
            if neighbor not in visited and len(visited) < MAX_FRAGMENT_NODES:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    return visited


def _build_graph_fragment(
    graph: nx.DiGraph,
    subject_id: Any,
    hop_depth: int,
) -> Any:
    """Assemble a protobuf GraphFragment from the in-memory graph.

    Performs BFS from subject_id, collects all transactions and nodes
    within hop_depth, converts to Purple's protobuf schema.
    """
    if pb2 is None:
        raise HTTPException(status_code=500, detail="Protobuf module not available")

    reachable_nodes = _bfs_subgraph(graph, subject_id, hop_depth)

    proto_transactions = []
    tx_count = 0

    edge_iter = (
        graph.edges(keys=True, data=True)
        if isinstance(graph, nx.MultiDiGraph)
        else ((u, v, None, d) for u, v, d in graph.edges(data=True))
    )

    for edge_tuple in edge_iter:
        if isinstance(graph, nx.MultiDiGraph):
            u, v, _key, data = edge_tuple
        else:
            u, v, _key, data = edge_tuple

        if u not in reachable_nodes and v not in reachable_nodes:
            continue
        if tx_count >= MAX_FRAGMENT_TRANSACTIONS:
            break

        tx_id = data.get("transaction_id", f"txn_{u}_{v}")
        amount = float(data.get("amount", 0.0))
        currency = data.get("currency", "USD")
        ts = _datetime_to_epoch(data.get("timestamp", 0))
        tx_type = str(data.get("transaction_type", "WIRE")).upper()
        reference = str(data.get("reference", ""))
        branch_code = str(data.get("branch_code", ""))

        proto_tx = pb2.Transaction(
            id=str(tx_id),
            source_node=str(u),
            target_node=str(v),
            amount=amount,
            currency=currency,
            timestamp=ts,
            type=tx_type,
            reference=reference,
            branch_code=branch_code,
        )
        proto_transactions.append(proto_tx)
        tx_count += 1

    proto_nodes: Dict[str, Any] = {}
    for node_id in sorted(reachable_nodes, key=str):
        node_data = graph.nodes.get(node_id, {})
        sid = str(node_id)
        risk_raw = node_data.get("risk_score", 0.5)
        risk_rating = "high" if risk_raw > 0.7 else ("medium" if risk_raw > 0.3 else "low")
        proto_nodes[sid] = pb2.NodeAttributes(
            id=sid,
            name=node_data.get("name", f"Entity {node_id}"),
            entity_type=node_data.get("entity_type", "individual"),
            jurisdiction=node_data.get("country", "US"),
            account_id=sid,
            ifsc_code=node_data.get("ifsc_code", ""),
            pan_number=node_data.get("pan_number", ""),
            address=node_data.get("address", ""),
            risk_rating=risk_rating,
            swift_code=node_data.get("swift", ""),
        )

    proto_text_evidence = []
    for idx, doc in enumerate(_evidence_documents):
        entity_id = str(doc.get("subject_id", ""))
        if entity_id and entity_id not in {str(n) for n in reachable_nodes}:
            continue
        proto_text_evidence.append(pb2.TextEvidence(
            id=doc.get("document_id", f"ev_{idx}"),
            source_type=doc.get("document_type", "memo"),
            content=doc.get("body", doc.get("narrative", "")),
            associated_entity=entity_id,
            timestamp=_datetime_to_epoch(doc.get("date", 0)),
        ))

    ground_truth_criminals: List[str] = []
    for crime in _ground_truth.get("crimes", []):
        for node_id in crime.get("nodes_involved", []):
            sid = str(node_id)
            if sid not in ground_truth_criminals:
                ground_truth_criminals.append(sid)
    ground_truth_criminals = sorted(ground_truth_criminals)

    fragment = pb2.GraphFragment(
        scenario_id=f"green-{int(time.time())}",
        generated_at=int(time.time()),
        transactions=proto_transactions,
        nodes=proto_nodes,
        text_evidence=proto_text_evidence,
        ground_truth_criminals=ground_truth_criminals,
    )
    return fragment


@app.post("/a2a")
async def a2a_endpoint(http_request: Request):
    """A2A protocol endpoint: receive InvestigationRequest, return GraphFragment.

    Accepts protobuf (application/x-protobuf) or JSON.
    Returns protobuf by default, or JSON if Accept header requests it.
    """
    if _current_graph is None:
        raise HTTPException(
            status_code=400,
            detail="No graph loaded. Generate data first with --generate-on-startup.",
        )

    content_type = http_request.headers.get("content-type", "")
    body = await http_request.body()

    subject_id: str = ""
    hop_depth: int = DEFAULT_HOP_DEPTH

    if PROTOBUF_CONTENT_TYPE in content_type and PROTOBUF_AVAILABLE and pb2 is not None:
        req = pb2.InvestigationRequest()
        req.ParseFromString(body)
        subject_id = req.subject_id
        hop_depth = req.hop_depth if req.hop_depth > 0 else DEFAULT_HOP_DEPTH
        logger.info(
            "A2A request (protobuf): subject_id=%s, hop_depth=%d, case_id=%s",
            subject_id, hop_depth, req.case_id,
        )
    else:
        import json as _json
        try:
            data = _json.loads(body)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid request body")
        subject_id = data.get("subject_id", "")
        hop_depth = int(data.get("hop_depth", DEFAULT_HOP_DEPTH))
        logger.info(
            "A2A request (JSON): subject_id=%s, hop_depth=%d",
            subject_id, hop_depth,
        )

    if not subject_id:
        raise HTTPException(status_code=422, detail="subject_id is required")

    resolved_id = _resolve_account_id(subject_id)
    if resolved_id not in _current_graph.nodes():
        raise HTTPException(
            status_code=404,
            detail=f"Subject node not found: {subject_id}",
        )

    fragment = _build_graph_fragment(_current_graph, resolved_id, hop_depth)

    accept = http_request.headers.get("accept", "")
    if "application/json" in accept and "application/x-protobuf" not in accept:
        fragment_dict = {
            "scenario_id": fragment.scenario_id,
            "generated_at": fragment.generated_at,
            "transactions": [
                {
                    "id": tx.id,
                    "source_node": tx.source_node,
                    "target_node": tx.target_node,
                    "amount": tx.amount,
                    "currency": tx.currency,
                    "timestamp": tx.timestamp,
                    "type": tx.type,
                    "reference": tx.reference,
                    "branch_code": tx.branch_code,
                }
                for tx in fragment.transactions
            ],
            "nodes": {
                nid: {
                    "id": na.id,
                    "name": na.name,
                    "entity_type": na.entity_type,
                    "jurisdiction": na.jurisdiction,
                    "account_id": na.account_id,
                    "address": na.address,
                    "risk_rating": na.risk_rating,
                    "swift_code": na.swift_code,
                }
                for nid, na in fragment.nodes.items()
            },
            "text_evidence": [
                {
                    "id": ev.id,
                    "source_type": ev.source_type,
                    "content": ev.content,
                    "associated_entity": ev.associated_entity,
                    "timestamp": ev.timestamp,
                }
                for ev in fragment.text_evidence
            ],
            "ground_truth_criminals": list(fragment.ground_truth_criminals),
        }
        return JSONResponse(content=fragment_dict)

    return FastAPIResponse(
        content=fragment.SerializeToString(),
        media_type=PROTOBUF_CONTENT_TYPE,
    )


@app.post("/results")
async def submit_results(http_request: Request):
    """Receive investigation results from Purple Agent.

    Accepts JSON with X-Idempotency-Key header for deduplication.
    Stores results and returns acknowledgment.
    """
    import json as _json

    idempotency_key = http_request.headers.get("X-Idempotency-Key", "")

    if idempotency_key and idempotency_key in _investigation_results:
        logger.info("Duplicate submission detected (key=%s), returning cached ack", idempotency_key)
        return JSONResponse(content={
            "status": "accepted",
            "duplicate": True,
            "idempotency_key": idempotency_key,
        })

    body = await http_request.body()
    try:
        result = _json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    result["received_at"] = int(time.time())
    result["idempotency_key"] = idempotency_key

    storage_key = idempotency_key or f"result-{int(time.time())}"
    _investigation_results[storage_key] = result

    logger.info(
        "Investigation result received: case_id=%s, typology=%s, entities=%d, confidence=%.2f",
        result.get("case_id", "?"),
        result.get("typology_detected", "?"),
        len(result.get("involved_entities", [])),
        result.get("confidence_score", 0.0),
    )

    return JSONResponse(content={
        "status": "accepted",
        "duplicate": False,
        "idempotency_key": idempotency_key,
        "case_id": result.get("case_id", ""),
    })


# Investigation Assessment Endpoint
@app.post("/a2a/investigation_assessment", response_model=InvestigationAssessmentResponse)
async def investigation_assessment(
    request: InvestigationAssessmentRequest
) -> InvestigationAssessmentResponse:
    """
    Evaluate a participant's investigation and return scored assessment with EFFICIENCY.
    
    This endpoint compares the participant's findings against ground truth
    and provides detailed feedback including efficiency metrics.
    
    The assessment evaluates:
    - Pattern identification (28%): How well did the participant identify crime patterns?
    - Evidence quality (20%): Quality and completeness of supporting evidence
    - Narrative clarity (16%): Clarity and structure of the investigation narrative
    - Completeness (16%): Coverage of all indicators in ground truth
    - Efficiency (20%): How few tool calls were needed
    """
    if not _ground_truth:
        raise HTTPException(
            status_code=400, 
            detail="No ground truth loaded. Generate and inject crimes first."
        )
    
    # Extract investigation data
    investigation_data = request.investigation_data or {}
    
    # Get crimes from ground truth
    actual_crimes = _ground_truth.get('crimes', [])
    identified_crimes = investigation_data.get('identified_crimes', [])
    
    # Calculate individual scores
    pattern_score = _calculate_pattern_score(identified_crimes, actual_crimes)
    evidence_score = _calculate_evidence_quality(investigation_data)
    narrative_score = _calculate_narrative_clarity(investigation_data)
    completeness_score = _calculate_completeness(investigation_data, _ground_truth)
    
    # Calculate efficiency score based on tool call count
    tool_count = get_tool_count(request.participant_id)
    
    # Efficiency scoring:
    # - Optimal: 10-50 calls = 100 points (excellent)
    # - Good: 51-100 calls = 80 points (good)
    # - Fair: 101-200 calls = 60 points (fair)
    # - Poor: 200+ calls = decreasing score (poor)
    if tool_count <= 50:
        efficiency_score = 100.0
        efficiency_rank = "excellent"
    elif tool_count <= 100:
        efficiency_score = 80.0
        efficiency_rank = "good"
    elif tool_count <= 200:
        efficiency_score = 60.0
        efficiency_rank = "fair"
    else:
        efficiency_score = max(40.0 - (tool_count - 200) * 0.1, 10.0)
        efficiency_rank = "poor"
    
    # Calculate overall score with efficiency (weighted average)
    weights = {
        'pattern': 0.28,      # Reduced from 0.35
        'evidence': 0.20,     # Reduced from 0.25
        'narrative': 0.16,    # Reduced from 0.20
        'completeness': 0.16, # Reduced from 0.20
        'efficiency': 0.20    # NEW
    }
    
    total_score = (
        pattern_score * weights['pattern'] +
        evidence_score * weights['evidence'] +
        narrative_score * weights['narrative'] +
        completeness_score * weights['completeness'] +
        efficiency_score * weights['efficiency']
    )
    total_score = round(total_score, 2)
    
    # Find missed indicators
    missed = _find_missed_indicators(investigation_data, _ground_truth)
    
    # Generate feedback with efficiency info
    feedback = _generate_feedback(total_score, missed)
    feedback += f"\n\nEfficiency: {tool_count} tool calls ({efficiency_rank})"
    
    return InvestigationAssessmentResponse(
        score=total_score,
        feedback=feedback,
        rubric_breakdown=RubricBreakdown(
            pattern_identification=pattern_score,
            evidence_quality=evidence_score,
            narrative_clarity=narrative_score,
            completeness=completeness_score
        ),
        missed_indicators=missed,
        tool_call_count=tool_count,
        efficiency_score=efficiency_score,
        efficiency_rank=efficiency_rank
    )


# A2A Tool Endpoints
@app.post("/a2a/tools/get_transactions", response_model=None)
async def get_transactions(
    request: GetTransactionsRequest,
    http_request: Request
):
    """
    Retrieve transaction history for an account.
    
    Returns all transactions where the account is either source or target,
    filtered by the specified parameters.
    
    Supports Protobuf serialization when Accept: application/x-protobuf header is set.
    """
    if _current_graph is None:
        raise HTTPException(
            status_code=400, 
            detail="No graph loaded. Generate a graph first."
        )
    
    account_id = _resolve_account_id(request.account_id)
    
    if account_id not in _current_graph.nodes():
        raise HTTPException(
            status_code=404, 
            detail=f"Account not found: {request.account_id}"
        )
    
    transactions = []
    
    # Get outbound transactions
    if request.direction in [TransactionDirection.OUTBOUND, TransactionDirection.BOTH]:
        for _, target, data in _current_graph.out_edges(account_id, data=True):
            txn = _edge_to_transaction(account_id, target, data)
            if _filter_transaction(txn, request):
                transactions.append(txn)
    
    # Get inbound transactions
    if request.direction in [TransactionDirection.INBOUND, TransactionDirection.BOTH]:
        for source, _, data in _current_graph.in_edges(account_id, data=True):
            txn = _edge_to_transaction(source, account_id, data)
            if _filter_transaction(txn, request):
                transactions.append(txn)
    
    # Sort by timestamp and apply limit
    transactions.sort(key=lambda t: t.timestamp, reverse=True)
    transactions = transactions[:request.limit]
    
    response_model = TransactionListResponse(
        account_id=request.account_id,
        transaction_count=len(transactions),
        transactions=transactions
    )
    
    # Check if Protobuf requested
    accept = http_request.headers.get("accept", "")
    if (
        "application/x-protobuf" in accept 
        and PROTOBUF_AVAILABLE 
        and pb2 is not None
        and hasattr(pb2, "LegacyTransactionListResponse")
        and hasattr(pb2, "LegacyTransaction")
    ):
        # Convert to Protobuf
        proto_response = pb2.LegacyTransactionListResponse(  # type: ignore[attr-defined]
            account_id=response_model.account_id,
            transaction_count=response_model.transaction_count,
            transactions=[
                pb2.LegacyTransaction(  # type: ignore[attr-defined]
                    transaction_id=txn.transaction_id,
                    source=txn.source,
                    target=txn.target,
                    amount=txn.amount,
                    currency=txn.currency,
                    timestamp=txn.timestamp.isoformat(),
                    transaction_type=txn.transaction_type.value,
                    memo=txn.memo or ""
                )
                for txn in response_model.transactions
            ]
        )
        
        # Return binary response
        return FastAPIResponse(
            content=proto_response.SerializeToString(),
            media_type="application/x-protobuf"
        )
    
    # Return JSON (default)
    return response_model


@app.post("/a2a/tools/get_kyc_profile", response_model=KycProfileResponse)
async def get_kyc_profile(request: GetKycProfileRequest) -> KycProfileResponse:
    """
    Retrieve KYC profile for an account.
    
    Returns entity information, risk score, and verification status.
    """
    if _current_graph is None:
        raise HTTPException(
            status_code=400, 
            detail="No graph loaded. Generate a graph first."
        )
    
    account_id = _resolve_account_id(request.account_id)
    
    if account_id not in _current_graph.nodes():
        raise HTTPException(
            status_code=404, 
            detail=f"Account not found: {request.account_id}"
        )
    
    node_data = _current_graph.nodes[account_id]
    
    return KycProfileResponse(
        account_id=request.account_id,
        entity_type=EntityType(node_data.get("entity_type", "person")),
        name=node_data.get("name", "Unknown"),
        company=node_data.get("company"),
        address=node_data.get("address"),
        country=node_data.get("country"),
        swift_code=node_data.get("swift"),
        iban=node_data.get("iban"),
        risk_score=node_data.get("risk_score", 0.5),
        verification_status=VerificationStatus(
            node_data.get("verification_status", "verified")
        ),
        created_at=node_data.get("created_at"),
        last_activity=node_data.get("last_activity")
    )


@app.post("/a2a/tools/get_account_connections", response_model=AccountConnectionsResponse)
async def get_account_connections(
    request: GetAccountConnectionsRequest
) -> AccountConnectionsResponse:
    """
    Retrieve accounts connected through transactions.
    
    Returns a list of connected accounts with relationship details.
    """
    if _current_graph is None:
        raise HTTPException(
            status_code=400, 
            detail="No graph loaded. Generate a graph first."
        )
    
    account_id = _resolve_account_id(request.account_id)
    
    if account_id not in _current_graph.nodes():
        raise HTTPException(
            status_code=404, 
            detail=f"Account not found: {request.account_id}"
        )
    
    connections: Dict[str, AccountConnection] = {}
    
    # Get direct connections (depth=1)
    # Outbound connections (this account is sender)
    for _, target, data in _current_graph.out_edges(account_id, data=True):
        if target not in connections:
            connections[target] = AccountConnection(
                account_id=str(target),
                relationship=ConnectionRelationship.RECEIVER,
                transaction_count=0,
                total_amount=0.0
            )
        connections[target].transaction_count += 1
        connections[target].total_amount += data.get("amount", 0.0)
    
    # Inbound connections (this account is receiver)
    for source, _, data in _current_graph.in_edges(account_id, data=True):
        if source not in connections:
            connections[source] = AccountConnection(
                account_id=str(source),
                relationship=ConnectionRelationship.SENDER,
                transaction_count=0,
                total_amount=0.0
            )
        else:
            # Already exists as receiver, so now it's both
            connections[source].relationship = ConnectionRelationship.BOTH
        connections[source].transaction_count += 1
        connections[source].total_amount += data.get("amount", 0.0)
    
    # TODO: Implement depth > 1 traversal if needed
    
    connection_list = list(connections.values())
    
    return AccountConnectionsResponse(
        account_id=request.account_id,
        connection_count=len(connection_list),
        connections=connection_list
    )


@app.post("/a2a/tools/get_evidence", response_model=EvidenceListResponse)
async def get_evidence(request: GetEvidenceRequest) -> EvidenceListResponse:
    """
    Retrieve evidence documents (SARs, emails, receipts).
    
    This is the "Sherlock Holmes" tool. The Purple Agent must:
    1. Read the documents to find entity IDs
    2. Extract the IDs
    3. Query the graph with those IDs
    
    This forces cognitive reasoning, not just SQL queries.
    """
    matching_docs: List[EvidenceDocument] = []
    
    for doc in _evidence_documents:
        # Filter by entity_id
        if request.entity_id and doc.get('subject_id') != request.entity_id:
            continue
        
        # Filter by document_type
        if request.document_type and doc.get('document_type') != request.document_type:
            continue
        
        # Search by keyword
        if request.contains_keyword:
            searchable_text = doc.get('body', '') + doc.get('narrative', '')
            if request.contains_keyword.lower() not in searchable_text.lower():
                continue
        
        # Build response document
        try:
            doc_date = datetime.fromisoformat(doc.get('date', datetime.now().isoformat()))
        except (ValueError, TypeError):
            doc_date = datetime.now()
        
        evidence_doc = EvidenceDocument(
            document_id=f"doc_{len(matching_docs)}",
            document_type=doc.get('document_type', 'unknown'),
            subject_id=doc.get('subject_id'),
            date=doc_date,
            content=doc.get('body', doc.get('narrative', '')),
            metadata={k: v for k, v in doc.items() if k not in ['body', 'narrative', 'date']}
        )
        matching_docs.append(evidence_doc)
        
        if len(matching_docs) >= request.limit:
            break
    
    return EvidenceListResponse(
        document_count=len(matching_docs),
        documents=matching_docs
    )


# =============================================================================
# Assessment Helper Functions
# =============================================================================

def _calculate_pattern_score(
    identified_crimes: List[Dict[str, Any]], 
    actual_crimes: List[Dict[str, Any]]
) -> float:
    """
    Calculate score for pattern identification (precision/recall based).
    
    Args:
        identified_crimes: List of crimes identified by the participant
        actual_crimes: List of actual crimes in ground truth
    
    Returns:
        Score from 0 to 100
    """
    if not actual_crimes:
        return 100.0 if not identified_crimes else 50.0
    
    if not identified_crimes:
        return 0.0
    
    # Extract crime types and nodes from identified crimes
    identified_set = set()
    for crime in identified_crimes:
        crime_type = crime.get('crime_type', '')
        nodes = tuple(sorted(crime.get('nodes', [])))
        identified_set.add((crime_type, nodes))
    
    # Extract crime types and nodes from actual crimes
    actual_set = set()
    for crime in actual_crimes:
        crime_type = crime.get('crime_type', '')
        nodes = tuple(sorted(crime.get('nodes_involved', [])))
        actual_set.add((crime_type, nodes))
    
    # Calculate precision and recall
    true_positives = len(identified_set & actual_set)
    precision = true_positives / len(identified_set) if identified_set else 0
    recall = true_positives / len(actual_set) if actual_set else 0
    
    # F1 score scaled to 100
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1 * 100, 2)


def _calculate_evidence_quality(investigation_data: Dict[str, Any]) -> float:
    """
    Calculate score for evidence quality.
    
    Args:
        investigation_data: Investigation findings from participant
    
    Returns:
        Score from 0 to 100
    """
    score = 0.0
    max_score = 100.0
    
    # Check for transaction evidence
    if investigation_data.get('transaction_ids'):
        score += 25.0
    
    # Check for account evidence
    if investigation_data.get('suspicious_accounts'):
        score += 25.0
    
    # Check for temporal analysis
    if investigation_data.get('temporal_patterns'):
        score += 25.0
    
    # Check for amount analysis
    if investigation_data.get('amount_patterns'):
        score += 25.0
    
    return min(score, max_score)


def _calculate_narrative_clarity(investigation_data: Dict[str, Any]) -> float:
    """
    Calculate score for narrative clarity.
    
    Args:
        investigation_data: Investigation findings from participant
    
    Returns:
        Score from 0 to 100
    """
    score = 0.0
    
    narrative = investigation_data.get('narrative', '')
    
    if not narrative:
        return 0.0
    
    # Check narrative length (reasonable detail)
    if len(narrative) >= 100:
        score += 30.0
    elif len(narrative) >= 50:
        score += 15.0
    
    # Check for key terminology
    key_terms = ['structuring', 'layering', 'suspicious', 'threshold', 
                 'pattern', 'transaction', 'account', 'transfer']
    terms_found = sum(1 for term in key_terms if term.lower() in narrative.lower())
    score += min(terms_found * 10, 40)
    
    # Check for structured sections
    if investigation_data.get('findings'):
        score += 15.0
    if investigation_data.get('recommendations'):
        score += 15.0
    
    return min(score, 100.0)


def _calculate_completeness(
    investigation_data: Dict[str, Any], 
    ground_truth: Dict[str, Any]
) -> float:
    """
    Calculate completeness score based on ground truth coverage.
    
    Args:
        investigation_data: Investigation findings from participant
        ground_truth: Actual crime data
    
    Returns:
        Score from 0 to 100
    """
    if not ground_truth:
        return 100.0
    
    score = 0.0
    total_checks = 0
    
    # Check if mule nodes were identified (for structuring)
    crimes = ground_truth.get('crimes', [])
    for crime in crimes:
        if crime.get('crime_type') == 'structuring':
            total_checks += 1
            mule_id = crime.get('metadata', {}).get('mule_id')
            suspicious_accounts = investigation_data.get('suspicious_accounts', [])
            if mule_id and str(mule_id) in [str(a) for a in suspicious_accounts]:
                score += 1
        
        if crime.get('crime_type') == 'layering':
            total_checks += 1
            chain_nodes = crime.get('nodes_involved', [])
            suspicious_accounts = investigation_data.get('suspicious_accounts', [])
            # Check if any chain nodes were identified
            identified = sum(1 for n in chain_nodes if str(n) in [str(a) for a in suspicious_accounts])
            if identified > 0:
                score += identified / len(chain_nodes) if chain_nodes else 0
    
    if total_checks == 0:
        return 100.0
    
    return round((score / total_checks) * 100, 2)


def _find_missed_indicators(
    investigation_data: Dict[str, Any], 
    ground_truth: Dict[str, Any]
) -> List[str]:
    """
    Identify indicators that were missed by the participant.
    
    Args:
        investigation_data: Investigation findings from participant
        ground_truth: Actual crime data
    
    Returns:
        List of missed indicator descriptions
    """
    missed = []
    
    crimes = ground_truth.get('crimes', [])
    identified_crimes = investigation_data.get('identified_crimes', [])
    identified_types = {c.get('crime_type') for c in identified_crimes}
    
    for crime in crimes:
        crime_type = crime.get('crime_type', '')
        metadata = crime.get('metadata', {})
        
        if crime_type == 'structuring' and 'structuring' not in identified_types:
            mule_id = metadata.get('mule_id')
            source_count = metadata.get('source_count', 0)
            missed.append(
                f"Structuring pattern: {source_count} sources to mule node {mule_id}"
            )
        
        if crime_type == 'layering' and 'layering' not in identified_types:
            chain_length = metadata.get('chain_length', 0)
            initial = metadata.get('initial_amount', 0)
            final = metadata.get('final_amount', 0)
            missed.append(
                f"Layering chain: {chain_length} hops, ${initial:,.2f} -> ${final:,.2f}"
            )
    
    # Check for missed temporal patterns
    if not investigation_data.get('temporal_patterns'):
        for crime in crimes:
            if crime.get('crime_type') == 'structuring':
                missed.append("Temporal clustering: multiple transfers within 48-hour window")
                break
    
    # Check for missed amount patterns
    if not investigation_data.get('amount_patterns'):
        for crime in crimes:
            if crime.get('crime_type') == 'structuring':
                missed.append("Amount pattern: transactions just below $10,000 CTR threshold")
                break
    
    return missed


def _generate_feedback(score: float, missed_indicators: List[str]) -> str:
    """
    Generate human-readable feedback based on score and missed indicators.
    
    Args:
        score: Overall investigation score (0-100)
        missed_indicators: List of missed indicator descriptions
    
    Returns:
        Feedback string
    """
    if score >= 90:
        feedback = "Excellent investigation. Comprehensive identification of financial crime patterns."
    elif score >= 70:
        feedback = "Good investigation with solid pattern identification. Some areas for improvement."
    elif score >= 50:
        feedback = "Moderate investigation. Several key indicators were missed."
    elif score >= 30:
        feedback = "Investigation needs significant improvement. Many patterns went undetected."
    else:
        feedback = "Investigation requires substantial enhancement. Consider reviewing AML typologies."
    
    if missed_indicators:
        feedback += f" Missed {len(missed_indicators)} indicator(s)."
    
    return feedback


# =============================================================================
# Transaction Helper Functions
# =============================================================================

def _edge_to_transaction(source: str, target: str, data: Dict[str, Any]) -> Transaction:
    """Convert graph edge to Transaction model."""
    return Transaction(
        transaction_id=data.get("transaction_id", str(uuid.uuid4())),
        source=str(source),
        target=str(target),
        amount=data.get("amount", 0.0),
        currency=data.get("currency", "USD"),
        timestamp=data.get("timestamp", datetime.now()),
        transaction_type=TransactionType(data.get("transaction_type", "wire")),
        memo=data.get("memo")
    )


def _filter_transaction(
    txn: Transaction, 
    request: GetTransactionsRequest
) -> bool:
    """Check if transaction passes request filters."""
    if request.start_date and txn.timestamp < request.start_date:
        return False
    if request.end_date and txn.timestamp > request.end_date:
        return False
    return True


def create_agent_json() -> dict:
    """Create agent.json manifest file content."""
    return AgentManifest().model_dump()


__all__ = [
    'app',
    'AgentManifest',
    'HealthResponse',
    'InvestigationAssessmentRequest',
    'InvestigationAssessmentResponse',
    'GetTransactionsRequest',
    'TransactionListResponse',
    'Transaction',
    'GetKycProfileRequest',
    'KycProfileResponse',
    'GetAccountConnectionsRequest',
    'AccountConnectionsResponse',
    'GetEvidenceRequest',
    'EvidenceDocument',
    'EvidenceListResponse',
    'set_graph',
    'set_ground_truth',
    'set_evidence',
    'create_agent_json',
    'PROTOBUF_AVAILABLE',
    'reset_tool_counter',
    'increment_tool_counter',
    'get_tool_count',
]
