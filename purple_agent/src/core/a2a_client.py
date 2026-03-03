"""
A2A Client -- Protobuf-enabled communication with Green Agent
PRD Reference: Task A4

v12.0 FIXES:
- [HIGH-21] _protobuf_to_dict: validates source_node/target_node non-empty (warns on empty)
- [LOW-17] fetch_graph: dead `import json` removed

v11.0 FIXES (retained):
- [HIGH-18] _protobuf_to_dict: validates transaction timestamps (warns on <=0)

v10.0 FIXES (retained):
- [HIGH-16] _validate_graph_fragment: dedicated warning for empty nodes dict
- [MED-20] _protobuf_to_dict: validates text_evidence id/content
- [MED-21] _retry_with_backoff: logs CircuitBreakerOpen before re-raise
- [LOW-08] CircuitBreaker: MIN_CALLS_BEFORE_TRIP extracted to constant

All v9.0 fixes retained: CRIT-07, HIGH-07, HIGH-08, HIGH-12, no quantization, Step 2.5
"""
import asyncio
import hashlib
import random
import time
import logging
from decimal import Decimal
from typing import Any

import httpx

from src.config import (
    RETRY_MAX_ATTEMPTS,
    RETRY_BASE_DELAY_SECONDS,
    JITTER_FACTOR,
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_WINDOW_SECONDS,
    PROTOBUF_CONTENT_TYPE,
    JSON_CONTENT_TYPE,
    GREEN_AGENT_URL,
    REQUEST_TIMEOUT_SECONDS,
    IDEMPOTENCY_HASH_ALGO,
)
from protos import financial_crime_pb2 as pb2

logger = logging.getLogger(__name__)

_CIRCUIT_BREAKER_MIN_CALLS: int = 5


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is in open state."""
    pass


class CircuitBreaker:
    """Simple circuit breaker with failure rate tracking.

    Design note: record_success unconditionally closes the breaker.
    Intentional -- Ralph kills processes on failure, so half-open is unnecessary.
    """

    def __init__(self, failure_threshold: float, window_seconds: int) -> None:
        self.failure_threshold = failure_threshold
        self.window_seconds = window_seconds
        self._successes: int = 0
        self._failures: int = 0
        self._window_start: float = time.time()
        self._is_open: bool = False

    def record_success(self) -> None:
        self._reset_window_if_expired()
        self._successes += 1
        self._is_open = False

    def record_failure(self) -> None:
        self._reset_window_if_expired()
        self._failures += 1
        total = self._successes + self._failures
        if (
            total >= _CIRCUIT_BREAKER_MIN_CALLS
            and (self._failures / total) >= self.failure_threshold
        ):
            self._is_open = True
            logger.warning(
                "Circuit breaker OPENED: %d/%d failures (%.0f%%) >= %.0f%%",
                self._failures, total,
                (self._failures / total) * 100,
                self.failure_threshold * 100,
            )

    def check(self) -> None:
        self._reset_window_if_expired()
        if self._is_open:
            raise CircuitBreakerOpen(
                "Circuit breaker is OPEN -- too many failures"
            )

    def _reset_window_if_expired(self) -> None:
        if time.time() - self._window_start > self.window_seconds:
            self._successes = 0
            self._failures = 0
            self._window_start = time.time()
            self._is_open = False


def _validate_amount(tx_id: str, raw_float: float) -> Decimal:
    """Convert Protobuf double to validated Decimal.

    Rule 3: NO QUANTIZATION AT INGESTION.
    v9.0 [CRIT-07]: Rejects zero-amount (strict positive).
    """
    raw_decimal = Decimal(str(raw_float))
    if not raw_decimal.is_finite():
        raise ValueError(
            f"Transaction {tx_id}: corrupted amount detected ({raw_decimal}). "
            f"NaN/Infinity values are rejected at the ingestion boundary."
        )
    if raw_decimal <= Decimal("0"):
        raise ValueError(
            f"Transaction {tx_id}: amount must be strictly positive, got {raw_decimal}"
        )
    return raw_decimal


def _validate_graph_fragment(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and sanitize a graph fragment per Step 2.5."""
    transactions = data.get("transactions", [])
    if not transactions:
        raise ValueError(
            "GraphFragment validation failed: zero transactions received. "
            "Empty graph = FAIL (Rule 19). Cannot investigate empty graph."
        )

    known_nodes = set(data.get("nodes", {}).keys())
    has_node_metadata = bool(known_nodes)

    if not has_node_metadata and transactions:
        logger.warning(
            "GraphFragment contains %d transaction(s) but no node metadata. "
            "Cannot validate source_node/target_node references.",
            len(transactions),
        )

    seen_ids: set[str] = set()
    deduped_transactions: list[dict[str, Any]] = []
    duplicate_count = 0

    for tx in transactions:
        tx_id = tx.get("id", "")
        if tx_id in seen_ids:
            duplicate_count += 1
            logger.warning(
                "Duplicate transaction ID '%s' found. Keeping first "
                "occurrence, dropping this duplicate (Rule 27).",
                tx_id,
            )
            continue
        seen_ids.add(tx_id)

        currency = tx.get("currency", "")
        if not currency or not currency.strip():
            raise ValueError(
                f"Transaction {tx_id}: currency field is empty. "
                f"Every transaction must have a valid ISO 4217 currency code."
            )

        if tx.get("source_node") == tx.get("target_node"):
            logger.warning(
                "Self-loop detected: transaction %s has "
                "source_node == target_node == '%s'. "
                "Kept for analysis (may be intra-bank transfer).",
                tx_id, tx.get("source_node"),
            )

        if has_node_metadata:
            for field in ("source_node", "target_node"):
                node_ref = tx.get(field, "")
                if node_ref and node_ref not in known_nodes:
                    logger.warning(
                        "Transaction %s: %s='%s' not found in nodes dict. "
                        "NetworkX will create an attribute-less node.",
                        tx_id, field, node_ref,
                    )

        deduped_transactions.append(tx)

    if duplicate_count > 0:
        logger.info(
            "GraphFragment validation: removed %d duplicate transaction(s). "
            "%d unique transactions remain.",
            duplicate_count, len(deduped_transactions),
        )

    return {**data, "transactions": deduped_transactions}


def _protobuf_to_dict(fragment: Any) -> dict[str, Any]:
    """Convert Protobuf GraphFragment to Python dict.

    v12.0 [HIGH-21]: Validates source_node/target_node non-empty (warns).
    v11.0 [HIGH-18]: Validates transaction timestamps (warns on <=0).
    v10.0 [MED-20]: Validates text_evidence id/content non-empty.
    v8.0+ [HIGH-07]: Validates currency BEFORE defaulting.
    """
    transactions: list[dict[str, Any]] = []
    for tx in fragment.transactions:
        amount = _validate_amount(tx.id, tx.amount)

        if not tx.currency or not tx.currency.strip():
            raise ValueError(
                f"Transaction {tx.id}: empty currency in Protobuf payload. "
                f"Every transaction must have a valid ISO 4217 currency code."
            )

        if tx.timestamp <= 0:
            logger.warning(
                "Transaction %s: timestamp=%d is non-positive. "
                "Expected Unix epoch > 0. Layering hop delay calculations "
                "may be incorrect.",
                tx.id, tx.timestamp,
            )

        if not tx.source_node or not tx.source_node.strip():
            logger.warning(
                "Transaction %s: source_node is empty. "
                "NetworkX will create an attribute-less '' node.",
                tx.id,
            )

        if not tx.target_node or not tx.target_node.strip():
            logger.warning(
                "Transaction %s: target_node is empty. "
                "NetworkX will create an attribute-less '' node.",
                tx.id,
            )

        transactions.append({
            "id": tx.id,
            "source_node": tx.source_node,
            "target_node": tx.target_node,
            "amount": amount,
            "currency": tx.currency,
            "timestamp": tx.timestamp,
            "type": tx.type or "WIRE",
            "reference": tx.reference,
            "branch_code": tx.branch_code,
        })

    nodes: dict[str, dict[str, Any]] = {}
    for node_id, node_attrs in fragment.nodes.items():
        nodes[node_id] = {
            "id": node_attrs.id,
            "name": node_attrs.name,
            "entity_type": node_attrs.entity_type,
            "jurisdiction": node_attrs.jurisdiction,
            "account_id": node_attrs.account_id,
            "ifsc_code": node_attrs.ifsc_code,
            "pan_number": node_attrs.pan_number,
            "address": node_attrs.address,
            "risk_rating": node_attrs.risk_rating,
            "swift_code": node_attrs.swift_code,
        }

    text_evidence: list[dict[str, Any]] = []
    for ev in fragment.text_evidence:
        if not ev.id:
            logger.warning(
                "Text evidence with empty id found (content starts with: "
                "'%s...'). Cannot be cited in SAR narrative.",
                ev.content[:50],
            )
        if not ev.content or not ev.content.strip():
            logger.warning(
                "Text evidence %s: content is empty. "
                "Evidence with no text provides zero analytical value.",
                ev.id or "(no id)",
            )
        text_evidence.append({
            "id": ev.id,
            "source_type": ev.source_type,
            "content": ev.content,
            "associated_entity": ev.associated_entity,
            "timestamp": ev.timestamp,
        })

    ground_truth = list(fragment.ground_truth_criminals)

    raw_data: dict[str, Any] = {
        "transactions": transactions,
        "nodes": nodes,
        "text_evidence": text_evidence,
        "ground_truth_criminals": ground_truth,
    }
    return _validate_graph_fragment(raw_data)


class A2AClient:
    """Communicates with Green Agent via A2A protocol."""

    def __init__(self, green_agent_url: str = GREEN_AGENT_URL) -> None:
        self.green_agent_url = green_agent_url
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            window_seconds=CIRCUIT_BREAKER_WINDOW_SECONDS,
        )
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT_SECONDS,
                headers={"Accept": PROTOBUF_CONTENT_TYPE},
            )
        return self._client

    async def _retry_with_backoff(
        self,
        coro_factory: Any,
        max_attempts: int = RETRY_MAX_ATTEMPTS,
    ) -> Any:
        """Retry with exponential backoff + jitter."""
        if max_attempts <= 0:
            raise ValueError("max_attempts must be > 0")

        last_exception: Exception | None = None
        for attempt in range(max_attempts):
            try:
                self.circuit_breaker.check()
                result = await coro_factory()
                self.circuit_breaker.record_success()
                return result
            except CircuitBreakerOpen as e:
                logger.error(
                    "Circuit breaker is OPEN on attempt %d/%d. "
                    "All communication with Green Agent is halted. Error: %s",
                    attempt + 1, max_attempts, e,
                )
                raise
            except Exception as e:
                last_exception = e
                self.circuit_breaker.record_failure()
                if attempt < max_attempts - 1:
                    delay = RETRY_BASE_DELAY_SECONDS * (2 ** attempt)
                    jitter = random.uniform(0, delay * JITTER_FACTOR)
                    await asyncio.sleep(delay + jitter)

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Retry loop exhausted with no exception captured")

    async def fetch_graph(
        self, subject_id: str, hop_depth: int = 3
    ) -> dict[str, Any]:
        """Fetch graph fragment from Green Agent."""
        client = await self._get_client()

        async def _do_fetch() -> dict[str, Any]:
            response = await client.post(
                f"{self.green_agent_url}/a2a",
                content=pb2.InvestigationRequest(
                    subject_id=subject_id, hop_depth=hop_depth,
                ).SerializeToString(),
                headers={"Content-Type": PROTOBUF_CONTENT_TYPE},
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if PROTOBUF_CONTENT_TYPE in content_type:
                fragment = pb2.GraphFragment()
                fragment.ParseFromString(response.content)
                return _protobuf_to_dict(fragment)
            else:
                return response.json()

        return await self._retry_with_backoff(_do_fetch)

    async def submit_result(
        self, result: dict[str, Any], case_id: str
    ) -> dict[str, Any]:
        """Submit investigation result with idempotency key.

        Rule 18: key = SHA-256(case_id + typology + sorted(involved_entities))
        """
        client = await self._get_client()

        typology = result.get("typology_detected", "NONE")
        entities = sorted(result.get("involved_entities", []))
        key_material = f"{case_id}:{typology}:{','.join(entities)}"
        idempotency_key = hashlib.new(
            IDEMPOTENCY_HASH_ALGO, key_material.encode()
        ).hexdigest()

        async def _do_submit() -> dict[str, Any]:
            response = await client.post(
                f"{self.green_agent_url}/results",
                json=result,
                headers={"X-Idempotency-Key": idempotency_key},
            )
            response.raise_for_status()
            return response.json()

        return await self._retry_with_backoff(_do_submit)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
