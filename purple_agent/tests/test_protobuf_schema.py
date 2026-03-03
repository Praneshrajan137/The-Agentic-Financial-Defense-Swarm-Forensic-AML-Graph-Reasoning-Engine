"""
Test: Protobuf Schema Round-Trip Serialization
PRD Reference: Task A3
"""
import time
import json
import pytest
from decimal import Decimal
from protos import financial_crime_pb2 as pb2


class TestProtobufSerialization:

    def test_transaction_round_trip(self):
        tx = pb2.Transaction(
            id="TX-001", source_node="alice", target_node="bob",
            amount=9500.0, currency="USD", timestamp=1700000000,
            type="WIRE", reference="REF-001",
        )
        data = tx.SerializeToString()
        tx2 = pb2.Transaction()
        tx2.ParseFromString(data)
        if tx2.id != "TX-001":
            raise ValueError(f"Expected id 'TX-001', got '{tx2.id}'")
        if Decimal(str(tx2.amount)) != Decimal("9500.0"):
            raise ValueError(f"Amount mismatch: {tx2.amount}")

    def test_graph_fragment_round_trip(self):
        fragment = pb2.GraphFragment(scenario_id="test-scenario")
        tx = fragment.transactions.add()
        tx.id = "TX-001"
        tx.amount = 9500.0
        node = fragment.nodes["alice"]
        node.id = "alice"
        node.name = "Alice"
        ev = fragment.text_evidence.add()
        ev.id = "EV-001"
        ev.content = "Test evidence"
        fragment.ground_truth_criminals.append("criminal_1")
        data = fragment.SerializeToString()
        f2 = pb2.GraphFragment()
        f2.ParseFromString(data)
        if len(f2.transactions) != 1:
            raise ValueError(f"Expected 1 transaction, got {len(f2.transactions)}")
        if "alice" not in f2.nodes:
            raise ValueError("Node 'alice' not found")
        if len(f2.text_evidence) != 1:
            raise ValueError(f"Expected 1 text evidence, got {len(f2.text_evidence)}")
        if "criminal_1" not in f2.ground_truth_criminals:
            raise ValueError("criminal_1 not in ground_truth")

    def test_protobuf_smaller_than_json(self):
        """Verify Protobuf encoding is significantly smaller than JSON."""
        fragment = pb2.GraphFragment(scenario_id="size-test")
        for i in range(100):
            tx = fragment.transactions.add()
            tx.id = f"TX-{i:04d}"
            tx.source_node = f"src_{i}"
            tx.target_node = f"tgt_{i}"
            tx.amount = 9500.0 + i
            tx.currency = "USD"
            tx.timestamp = 1700000000 + (i * 3600)
        proto_size = len(fragment.SerializeToString())
        json_size = len(json.dumps({
            "scenario_id": "size-test",
            "transactions": [
                {"id": f"TX-{i:04d}", "source_node": f"src_{i}",
                 "target_node": f"tgt_{i}", "amount": 9500.0 + i,
                 "currency": "USD", "timestamp": 1700000000 + (i * 3600)}
                for i in range(100)
            ],
        }).encode())
        ratio = proto_size / json_size
        if ratio >= 0.80:
            raise ValueError(
                f"Protobuf should be <80% of JSON size, got {ratio:.2%}"
            )

    def test_decimal_conversion_precision(self):
        tx = pb2.Transaction(amount=9999.99)
        data = tx.SerializeToString()
        tx2 = pb2.Transaction()
        tx2.ParseFromString(data)
        amount = Decimal(str(tx2.amount))
        if amount != Decimal("9999.99"):
            raise ValueError(f"Precision lost: {amount} != 9999.99")

    def test_decimal_conversion_stress(self):
        trouble_values = [0.1, 0.2, 100.1, 9999.99, 91267.30, 0.3]
        for val in trouble_values:
            tx = pb2.Transaction(amount=val)
            data = tx.SerializeToString()
            tx2 = pb2.Transaction()
            tx2.ParseFromString(data)
            converted = Decimal(str(tx2.amount))
            expected = Decimal(str(val)).quantize(Decimal("0.01"))
            actual = converted.quantize(Decimal("0.01"))
            if actual != expected:
                raise ValueError(
                    f"Stress test failed for {val}: {actual} != {expected}"
                )

    def test_investigation_request_round_trip(self):
        req = pb2.InvestigationRequest(
            subject_id="suspect_001", case_id="CASE-2026-001",
            hop_depth=3, jurisdiction="fincen",
        )
        data = req.SerializeToString()
        req2 = pb2.InvestigationRequest()
        req2.ParseFromString(data)
        if req2.subject_id != "suspect_001":
            raise ValueError(f"subject_id mismatch: {req2.subject_id}")
        if req2.hop_depth != 3:
            raise ValueError(f"hop_depth mismatch: {req2.hop_depth}")

    def test_hop_depth_zero_default(self):
        req = pb2.InvestigationRequest(subject_id="test")
        if req.hop_depth != 0:
            raise ValueError(f"Proto3 default should be 0, got {req.hop_depth}")

    def test_investigation_result_round_trip(self):
        result = pb2.InvestigationResult(
            case_id="CASE-001", sar_narrative="WHO: Subject...",
            typology_detected="STRUCTURING", confidence_score=0.95,
            jurisdiction="fincen", investigation_timestamp=int(time.time()),
        )
        result.involved_entities.extend(["e1", "e2"])
        evidence = result.cited_evidence.add()
        evidence.transaction_id = "TX-001"
        evidence.amount = 9500.0
        data = result.SerializeToString()
        r2 = pb2.InvestigationResult()
        r2.ParseFromString(data)
        if r2.typology_detected != "STRUCTURING":
            raise ValueError(f"typology mismatch: {r2.typology_detected}")
        if len(r2.involved_entities) != 2:
            raise ValueError(f"Expected 2 entities, got {len(r2.involved_entities)}")
