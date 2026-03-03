"""
Test: A2A Client -- Protobuf communication, retry, circuit breaker
PRD Reference: Task A4
28 tests covering protobuf conversion, validation, circuit breaker, and retry.
"""
import asyncio
import logging
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import math

import pytest
import httpx

from protos import financial_crime_pb2 as pb2
from src.core.a2a_client import (
    A2AClient,
    CircuitBreaker,
    CircuitBreakerOpen,
    _validate_amount,
    _validate_graph_fragment,
    _protobuf_to_dict,
    _CIRCUIT_BREAKER_MIN_CALLS,
)


# ═══════════════════════════════════════════════════════════════════
# Helper: build a minimal valid GraphFragment protobuf
# ═══════════════════════════════════════════════════════════════════

def _make_fragment(
    tx_id: str = "TX-001",
    amount: float = 9500.0,
    currency: str = "USD",
    source: str = "alice",
    target: str = "bob",
    timestamp: int = 1700000000,
    with_node: bool = True,
    with_evidence: bool = False,
    with_ground_truth: bool = False,
) -> pb2.GraphFragment:
    fragment = pb2.GraphFragment(scenario_id="test")
    tx = fragment.transactions.add()
    tx.id = tx_id
    tx.source_node = source
    tx.target_node = target
    tx.amount = amount
    tx.currency = currency
    tx.timestamp = timestamp
    tx.type = "WIRE"
    if with_node:
        for nid in (source, target):
            if nid:
                node = fragment.nodes[nid]
                node.id = nid
                node.name = f"Entity {nid}"
    if with_evidence:
        ev = fragment.text_evidence.add()
        ev.id = "EV-001"
        ev.content = "Test evidence content"
        ev.associated_entity = source
        ev.timestamp = 1700000100
    if with_ground_truth:
        fragment.ground_truth_criminals.append("criminal_1")
    return fragment


# ═══════════════════════════════════════════════════════════════════
# Tests 1-11: _protobuf_to_dict
# ═══════════════════════════════════════════════════════════════════

class TestProtobufToDict:

    def test_protobuf_to_dict_extracts_transactions(self):
        """Test 1: Transactions are extracted from protobuf."""
        fragment = _make_fragment()
        result = _protobuf_to_dict(fragment)
        if len(result["transactions"]) != 1:
            raise ValueError(f"Expected 1 tx, got {len(result['transactions'])}")
        if result["transactions"][0]["id"] != "TX-001":
            raise ValueError("TX id mismatch")

    def test_protobuf_to_dict_extracts_nodes(self):
        """Test 2: Nodes are extracted from protobuf."""
        fragment = _make_fragment()
        result = _protobuf_to_dict(fragment)
        if "alice" not in result["nodes"]:
            raise ValueError("Node 'alice' not found")
        if "bob" not in result["nodes"]:
            raise ValueError("Node 'bob' not found")

    def test_protobuf_to_dict_extracts_text_evidence(self):
        """Test 3: Text evidence is extracted."""
        fragment = _make_fragment(with_evidence=True)
        result = _protobuf_to_dict(fragment)
        if len(result["text_evidence"]) != 1:
            raise ValueError(f"Expected 1 evidence, got {len(result['text_evidence'])}")

    def test_protobuf_to_dict_extracts_ground_truth(self):
        """Test 4: Ground truth criminals are extracted."""
        fragment = _make_fragment(with_ground_truth=True)
        result = _protobuf_to_dict(fragment)
        if "criminal_1" not in result["ground_truth_criminals"]:
            raise ValueError("criminal_1 not found in ground truth")

    def test_protobuf_to_dict_amounts_are_decimal(self):
        """Test 5: Amounts are Decimal, not float."""
        fragment = _make_fragment(amount=9500.0)
        result = _protobuf_to_dict(fragment)
        amt = result["transactions"][0]["amount"]
        if not isinstance(amt, Decimal):
            raise TypeError(f"Amount should be Decimal, got {type(amt)}")

    def test_protobuf_to_dict_rejects_nan_amount(self):
        """Test 6: NaN amount is rejected."""
        fragment = _make_fragment(amount=float("nan"))
        with pytest.raises(ValueError, match="corrupted amount"):
            _protobuf_to_dict(fragment)

    def test_protobuf_to_dict_rejects_infinity_amount(self):
        """Test 7: Infinity amount is rejected."""
        fragment = _make_fragment(amount=float("inf"))
        with pytest.raises(ValueError, match="corrupted amount"):
            _protobuf_to_dict(fragment)

    def test_protobuf_to_dict_preserves_precision(self):
        """Test 8: No quantization at ingestion (Rule 3)."""
        fragment = _make_fragment(amount=91267.30)
        result = _protobuf_to_dict(fragment)
        amt = result["transactions"][0]["amount"]
        if amt != Decimal("91267.3"):
            raise ValueError(f"Precision lost: {amt}")

    def test_protobuf_to_dict_rejects_empty_currency(self):
        """Test 9: Empty currency in protobuf is rejected."""
        fragment = _make_fragment(currency="")
        with pytest.raises(ValueError, match="empty currency"):
            _protobuf_to_dict(fragment)

    def test_protobuf_to_dict_rejects_zero_amount(self):
        """Test 10: Zero amount is rejected (CRIT-07)."""
        fragment = pb2.GraphFragment(scenario_id="test")
        tx = fragment.transactions.add()
        tx.id = "TX-ZERO"
        tx.source_node = "a"
        tx.target_node = "b"
        tx.amount = 0.0
        tx.currency = "USD"
        tx.timestamp = 1700000000
        node_a = fragment.nodes["a"]
        node_a.id = "a"
        node_a.name = "A"
        with pytest.raises(ValueError, match="strictly positive"):
            _protobuf_to_dict(fragment)

    def test_protobuf_to_dict_warns_empty_evidence_id(self, caplog):
        """Test 11: Empty evidence id is warned, not rejected (MED-20)."""
        fragment = _make_fragment()
        ev = fragment.text_evidence.add()
        ev.id = ""
        ev.content = "Some content without an ID"
        ev.timestamp = 1700000100
        with caplog.at_level(logging.WARNING):
            result = _protobuf_to_dict(fragment)
        if len(result["text_evidence"]) != 1:
            raise ValueError("Evidence should be included despite empty id")
        if "empty id" not in caplog.text:
            raise ValueError("Expected warning about empty id")


# ═══════════════════════════════════════════════════════════════════
# Tests 12-17: _validate_graph_fragment
# ═══════════════════════════════════════════════════════════════════

class TestValidateGraphFragment:

    def test_validate_graph_fragment_rejects_empty(self):
        """Test 12: Empty graph is rejected (Rule 19)."""
        with pytest.raises(ValueError, match="zero transactions"):
            _validate_graph_fragment({"transactions": [], "nodes": {}})

    def test_validate_graph_fragment_deduplicates_txids(self, caplog):
        """Test 13: Duplicate tx IDs keep first, log warning (Rule 27)."""
        data = {
            "transactions": [
                {"id": "TX-1", "source_node": "a", "target_node": "b",
                 "amount": Decimal("100"), "currency": "USD", "timestamp": 1},
                {"id": "TX-1", "source_node": "c", "target_node": "d",
                 "amount": Decimal("200"), "currency": "USD", "timestamp": 2},
            ],
            "nodes": {},
        }
        with caplog.at_level(logging.WARNING):
            result = _validate_graph_fragment(data)
        if len(result["transactions"]) != 1:
            raise ValueError("Expected 1 tx after dedup")
        if result["transactions"][0]["source_node"] != "a":
            raise ValueError("First occurrence should be kept")
        if "Duplicate" not in caplog.text:
            raise ValueError("Expected duplicate warning")

    def test_validate_graph_fragment_logs_self_loops(self, caplog):
        """Test 14: Self-loops are logged, not rejected."""
        data = {
            "transactions": [
                {"id": "TX-1", "source_node": "a", "target_node": "a",
                 "amount": Decimal("100"), "currency": "USD", "timestamp": 1},
            ],
            "nodes": {},
        }
        with caplog.at_level(logging.WARNING):
            result = _validate_graph_fragment(data)
        if len(result["transactions"]) != 1:
            raise ValueError("Self-loop should be kept")
        if "Self-loop" not in caplog.text:
            raise ValueError("Expected self-loop warning")

    def test_validate_graph_fragment_rejects_empty_currency(self):
        """Test 15: Empty currency triggers ValueError."""
        data = {
            "transactions": [
                {"id": "TX-1", "source_node": "a", "target_node": "b",
                 "amount": Decimal("100"), "currency": "", "timestamp": 1},
            ],
            "nodes": {},
        }
        with pytest.raises(ValueError, match="currency"):
            _validate_graph_fragment(data)

    def test_validate_graph_fragment_warns_orphan_nodes(self, caplog):
        """Test 16: Orphan node references produce warning (HIGH-12)."""
        data = {
            "transactions": [
                {"id": "TX-1", "source_node": "known",
                 "target_node": "orphan",
                 "amount": Decimal("100"), "currency": "USD", "timestamp": 1},
            ],
            "nodes": {"known": {"id": "known", "name": "Known"}},
        }
        with caplog.at_level(logging.WARNING):
            _validate_graph_fragment(data)
        if "orphan" not in caplog.text.lower():
            raise ValueError("Expected orphan node warning")

    def test_validate_graph_fragment_warns_no_node_metadata(self, caplog):
        """Test 17: Empty nodes dict produces dedicated warning (HIGH-16)."""
        data = {
            "transactions": [
                {"id": "TX-1", "source_node": "a", "target_node": "b",
                 "amount": Decimal("100"), "currency": "USD", "timestamp": 1},
            ],
            "nodes": {},
        }
        with caplog.at_level(logging.WARNING):
            _validate_graph_fragment(data)
        if "no node metadata" not in caplog.text:
            raise ValueError("Expected no-node-metadata warning")


# ═══════════════════════════════════════════════════════════════════
# Tests 18-23: CircuitBreaker and Retry
# ═══════════════════════════════════════════════════════════════════

class TestCircuitBreakerAndRetry:

    def test_circuit_breaker_opens_on_failures(self):
        """Test 18: CB opens after threshold exceeded."""
        cb = CircuitBreaker(failure_threshold=0.6, window_seconds=60)
        for _ in range(_CIRCUIT_BREAKER_MIN_CALLS):
            cb.record_failure()
        with pytest.raises(CircuitBreakerOpen):
            cb.check()

    def test_circuit_breaker_resets_after_window(self):
        """Test 19: CB resets when window expires."""
        cb = CircuitBreaker(failure_threshold=0.6, window_seconds=1)
        for _ in range(_CIRCUIT_BREAKER_MIN_CALLS):
            cb.record_failure()
        cb._window_start = time.time() - 2
        cb.check()  # Should not raise after window expiry

    async def test_retry_with_backoff(self):
        """Test 20: Retry succeeds after transient failures."""
        client = A2AClient("http://fake:9090")
        call_count = 0

        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("Connection refused")
            return {"ok": True}

        result = await client._retry_with_backoff(flaky, max_attempts=5)
        if result != {"ok": True}:
            raise ValueError(f"Unexpected result: {result}")
        if call_count != 3:
            raise ValueError(f"Expected 3 calls, got {call_count}")

    async def test_retry_exhaustion_raises(self):
        """Test 21: All retries exhausted raises last exception."""
        client = A2AClient("http://fake:9090")

        async def always_fail():
            raise httpx.ConnectError("Connection refused")

        with pytest.raises(httpx.ConnectError):
            await client._retry_with_backoff(always_fail, max_attempts=2)

    async def test_retry_exhaustion_handles_zero_attempts(self):
        """Test 22: max_attempts=0 raises ValueError."""
        client = A2AClient("http://fake:9090")

        async def noop():
            return {}

        with pytest.raises(ValueError, match="max_attempts must be > 0"):
            await client._retry_with_backoff(noop, max_attempts=0)

    async def test_retry_logs_circuit_breaker_open(self, caplog):
        """Test 23: CircuitBreakerOpen is logged before re-raise (MED-21)."""
        client = A2AClient("http://fake:9090")
        cb = client.circuit_breaker
        for _ in range(_CIRCUIT_BREAKER_MIN_CALLS):
            cb.record_failure()

        async def noop():
            return {}

        with caplog.at_level(logging.ERROR):
            with pytest.raises(CircuitBreakerOpen):
                await client._retry_with_backoff(noop, max_attempts=3)
        if "OPEN" not in caplog.text:
            raise ValueError("Expected OPEN log message")


# ═══════════════════════════════════════════════════════════════════
# Tests 24-26: Submission and Fetch
# ═══════════════════════════════════════════════════════════════════

class TestSubmissionAndFetch:

    async def test_submit_result_includes_idempotency_key(self):
        """Test 24: Submit includes X-Idempotency-Key header (Rule 18)."""
        client = A2AClient("http://fake:9090")
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()

        captured_headers = {}

        async def mock_post(url, json=None, headers=None, **kwargs):
            captured_headers.update(headers or {})
            return mock_response

        mock_http = AsyncMock()
        mock_http.post = mock_post
        client._client = mock_http

        await client.submit_result(
            {"typology_detected": "STRUCTURING", "involved_entities": ["e1", "e2"]},
            case_id="CASE-001",
        )
        if "X-Idempotency-Key" not in captured_headers:
            raise ValueError("Missing X-Idempotency-Key header")

    async def test_fetch_graph_protobuf_content_type(self):
        """Test 25: Fetch sends protobuf content type."""
        client = A2AClient("http://fake:9090")

        fragment = _make_fragment()
        proto_bytes = fragment.SerializeToString()

        mock_response = MagicMock()
        mock_response.content = proto_bytes
        mock_response.headers = {"content-type": "application/x-protobuf"}
        mock_response.raise_for_status = MagicMock()

        captured_headers = {}

        async def mock_post(url, content=None, headers=None, **kwargs):
            captured_headers.update(headers or {})
            return mock_response

        mock_http = AsyncMock()
        mock_http.post = mock_post
        client._client = mock_http

        result = await client.fetch_graph("alice", hop_depth=3)
        if captured_headers.get("Content-Type") != "application/x-protobuf":
            raise ValueError("Expected protobuf content-type on request")
        if "transactions" not in result:
            raise ValueError("Result should contain transactions")

    async def test_fetch_graph_json_fallback(self):
        """Test 26: Falls back to JSON when response is not protobuf."""
        client = A2AClient("http://fake:9090")

        mock_response = MagicMock()
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {
            "transactions": [{"id": "TX-1"}],
            "nodes": {},
        }
        mock_response.raise_for_status = MagicMock()

        async def mock_post(url, content=None, headers=None, **kwargs):
            return mock_response

        mock_http = AsyncMock()
        mock_http.post = mock_post
        client._client = mock_http

        result = await client.fetch_graph("alice")
        if result["transactions"][0]["id"] != "TX-1":
            raise ValueError("JSON fallback should return parsed JSON")


# ═══════════════════════════════════════════════════════════════════
# Tests 27-28: v11.0 / v12.0 boundary validations
# ═══════════════════════════════════════════════════════════════════

class TestBoundaryValidations:

    def test_protobuf_to_dict_warns_zero_timestamp(self, caplog):
        """Test 27: Zero timestamp produces WARNING, not rejection (HIGH-18)."""
        fragment = _make_fragment(timestamp=0)
        with caplog.at_level(logging.WARNING):
            result = _protobuf_to_dict(fragment)
        if len(result["transactions"]) != 1:
            raise ValueError("Transaction should be kept despite zero timestamp")
        if "timestamp" not in caplog.text.lower():
            raise ValueError("Expected timestamp warning")

    def test_protobuf_to_dict_warns_empty_source_node(self, caplog):
        """Test 28: Empty source/target_node produces WARNING (HIGH-21)."""
        fragment = _make_fragment(source="", target="bob", with_node=False)
        node = fragment.nodes["bob"]
        node.id = "bob"
        node.name = "Bob"

        with caplog.at_level(logging.WARNING):
            result = _protobuf_to_dict(fragment)

        if len(result["transactions"]) != 1:
            raise ValueError("Transaction should be included despite empty source")
        if "source_node" not in caplog.text:
            raise ValueError("Expected source_node warning")

        # Also test empty target_node
        caplog.clear()
        fragment2 = _make_fragment(source="alice", target="", with_node=False)
        node2 = fragment2.nodes["alice"]
        node2.id = "alice"
        node2.name = "Alice"

        with caplog.at_level(logging.WARNING):
            result2 = _protobuf_to_dict(fragment2)

        if len(result2["transactions"]) != 1:
            raise ValueError("Transaction should be included despite empty target")
        if "target_node" not in caplog.text:
            raise ValueError("Expected target_node warning")
