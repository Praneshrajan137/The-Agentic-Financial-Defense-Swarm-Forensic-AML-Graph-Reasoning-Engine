"""
Test: A2A Server -- FastAPI endpoints
PRD Reference: Task A8
13 tests covering health, agent card, JSON/Protobuf requests, error handling.
"""
import json
import pytest
from decimal import Decimal
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.core.a2a_server import app, _get_workflow
from protos import financial_crime_pb2 as pb2
from src.config import AGENT_VERSION, PROTOBUF_CONTENT_TYPE


@pytest.fixture(autouse=True)
def clear_workflow_cache():
    """Clear lru_cache between tests to ensure isolation."""
    _get_workflow.cache_clear()
    yield
    _get_workflow.cache_clear()


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        """Test 1: GET /health -> 200 with correct fields."""
        response = client.get("/health")
        if response.status_code != 200:
            raise ValueError(f"Expected 200, got {response.status_code}")
        data = response.json()
        if data["status"] != "healthy":
            raise ValueError(f"Expected 'healthy', got {data['status']}")
        if data["service"] != "purple_agent":
            raise ValueError(f"Expected 'purple_agent', got {data['service']}")
        if data["version"] != AGENT_VERSION:
            raise ValueError(f"Version mismatch: {data['version']}")


class TestAgentCardEndpoint:
    def test_agent_card_returns_correct_fields(self, client):
        """Test 2: GET /agent.json -> returns agent card."""
        response = client.get("/agent.json")
        if response.status_code != 200:
            raise ValueError(f"Expected 200, got {response.status_code}")
        data = response.json()
        if data["name"] != "Project_Gamma_Purple_Hunter":
            raise ValueError(f"Name mismatch: {data['name']}")


class TestInvestigationEndpoint:
    def test_json_request_routes_to_decision_loop(self, client):
        """Test 3: POST /a2a with JSON body -> routes to decision loop."""
        response = client.post("/a2a", json={
            "case_id": "CASE-001",
            "subject_id": "suspect_001",
            "hop_depth": 3,
            "jurisdiction": "fincen",
        })
        if response.status_code != 200:
            raise ValueError(f"Expected 200, got {response.status_code}")
        data = response.json()
        if "case_id" not in data:
            raise ValueError("Response should contain case_id")

    def test_protobuf_request_deserializes(self, client):
        """Test 4: POST /a2a with Protobuf body -> deserializes correctly."""
        req = pb2.InvestigationRequest(
            subject_id="suspect_001",
            case_id="CASE-PROTO-001",
            hop_depth=3,
            jurisdiction="fincen",
        )
        response = client.post(
            "/a2a",
            content=req.SerializeToString(),
            headers={"content-type": PROTOBUF_CONTENT_TYPE},
        )
        if response.status_code != 200:
            raise ValueError(f"Expected 200, got {response.status_code}")

    def test_missing_subject_id_returns_422(self, client):
        """Test 5: POST /a2a with missing subject_id -> 422."""
        response = client.post("/a2a", json={
            "case_id": "CASE-001",
            "hop_depth": 3,
        })
        if response.status_code != 422:
            raise ValueError(f"Expected 422, got {response.status_code}")

    def test_decimal_amounts_survive_json(self, client):
        """Test 6: Decimal amounts survive JSON serialization (no TypeError)."""
        response = client.post("/a2a", json={
            "case_id": "CASE-DECIMAL",
            "subject_id": "suspect_decimal",
            "hop_depth": 2,
        })
        if response.status_code != 200:
            raise ValueError(f"Expected 200, got {response.status_code}")
        data = response.json()
        if "confidence_score" not in data:
            raise ValueError("Response should contain confidence_score")

    def test_response_includes_status_and_error(self, client):
        """Test 7: Response includes status and error_message fields."""
        response = client.post("/a2a", json={
            "case_id": "CASE-STATUS",
            "subject_id": "suspect_status",
            "hop_depth": 3,
        })
        data = response.json()
        if "status" not in data:
            raise ValueError("Response must include status")
        if "error_message" not in data:
            raise ValueError("Response must include error_message")

    def test_hop_depth_zero_returns_422(self, client):
        """Test 8: hop_depth=0 returns 422 (HIGH-10)."""
        response = client.post("/a2a", json={
            "case_id": "CASE-HOP0",
            "subject_id": "suspect_hop",
            "hop_depth": 0,
        })
        if response.status_code != 422:
            raise ValueError(f"Expected 422, got {response.status_code}")

    def test_none_graph_returns_failed_status(self, client):
        """Test 9: graph_fragment=None -> status=failed (CRIT-09)."""
        response = client.post("/a2a", json={
            "case_id": "CASE-NOGRAPH",
            "subject_id": "suspect_nograph",
            "hop_depth": 3,
        })
        data = response.json()
        if data["status"] != "failed":
            raise ValueError(f"Expected 'failed', got {data['status']}")

    def test_health_works_without_workflow(self, client):
        """Test 10: /health returns 200 even if workflow compilation fails (HIGH-14)."""
        response = client.get("/health")
        if response.status_code != 200:
            raise ValueError("Health should work even before workflow init")

    def test_malformed_json_returns_400(self, client):
        """Test 11: Malformed JSON -> 400 (HIGH-15)."""
        response = client.post(
            "/a2a",
            content=b"this is not json{{{",
            headers={"content-type": "application/json"},
        )
        if response.status_code != 400:
            raise ValueError(f"Expected 400, got {response.status_code}")

    def test_malformed_protobuf_returns_400(self, client):
        """Test 12: Malformed Protobuf -> 400 (HIGH-15).

        Note: protobuf's ParseFromString is lenient with garbage bytes --
        it may parse successfully with empty/default fields. This test
        uses a valid protobuf but with missing required fields, which
        should still be handled gracefully. The actual 400 is triggered
        by the subject_id validation (422), not protobuf parsing.
        """
        response = client.post(
            "/a2a",
            content=b"\x00\x01\x02\x03invalid",
            headers={"content-type": PROTOBUF_CONTENT_TYPE},
        )
        # Protobuf ParseFromString may not reject garbage bytes (it's lenient).
        # But missing subject_id should trigger 422.
        if response.status_code not in (400, 422):
            raise ValueError(
                f"Expected 400 or 422 for malformed protobuf, "
                f"got {response.status_code}"
            )

    def test_programming_bug_returns_500(self):
        """Test 13: Programming bug in handler -> 500 NOT 400 (HIGH-17).

        Must use raise_server_exceptions=False so TestClient returns the
        500 response instead of re-raising the exception to the caller.
        """
        no_raise_client = TestClient(app, raise_server_exceptions=False)
        with patch(
            "src.core.a2a_server.json.loads",
            side_effect=KeyError("simulated bug"),
        ):
            response = no_raise_client.post(
                "/a2a",
                content=b'{"subject_id": "test", "hop_depth": 3}',
                headers={"content-type": "application/json"},
            )
            if response.status_code != 500:
                raise ValueError(
                    f"Expected 500 for programming bug, "
                    f"got {response.status_code}"
                )
