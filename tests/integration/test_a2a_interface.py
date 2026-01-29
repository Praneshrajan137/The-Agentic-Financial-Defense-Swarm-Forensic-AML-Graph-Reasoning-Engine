"""
Test Suite for A2A Interface Module
===================================
Integration tests for src/core/a2a_interface.py FastAPI endpoints.

Tests cover:
1. Health endpoint
2. Agent manifest endpoint
3. get_transactions endpoint
4. get_kyc_profile endpoint
5. get_account_connections endpoint
6. investigation_assessment endpoint
7. Error handling (400, 404 responses)
"""

import pytest
from fastapi.testclient import TestClient
import networkx as nx

from src.core.a2a_interface import (
    app,
    set_graph,
    set_ground_truth,
    AgentManifest,
    HealthResponse
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def client():
    """Fixture providing TestClient without graph loaded."""
    # Reset state before each test
    set_graph(None)
    set_ground_truth({})
    return TestClient(app)


@pytest.fixture
def client_with_graph(baseline_graph):
    """Fixture providing TestClient with graph loaded."""
    set_graph(baseline_graph)
    set_ground_truth({"test": True})
    client = TestClient(app)
    yield client
    # Cleanup
    set_graph(None)
    set_ground_truth({})


# =============================================================================
# Tests for Health Endpoint
# =============================================================================

@pytest.mark.api
class TestHealthEndpoint:
    """Test suite for /health endpoint."""
    
    def test_health_returns_200(self, client):
        """Test that health endpoint returns 200 OK."""
        response = client.get("/health")
        
        assert response.status_code == 200
    
    def test_health_returns_healthy_status(self, client):
        """Test that health endpoint returns 'healthy' status."""
        response = client.get("/health")
        data = response.json()
        
        assert data["status"] == "healthy"
    
    def test_health_returns_version(self, client):
        """Test that health endpoint returns version."""
        response = client.get("/health")
        data = response.json()
        
        assert "version" in data
        assert data["version"] == "1.0.0"
    
    def test_health_shows_no_graph_loaded(self, client):
        """Test that health shows graph_loaded=False when no graph."""
        response = client.get("/health")
        data = response.json()
        
        assert data["graph_loaded"] is False
        assert data["node_count"] == 0
        assert data["edge_count"] == 0
    
    def test_health_shows_graph_loaded(self, client_with_graph):
        """Test that health shows graph info when loaded."""
        response = client_with_graph.get("/health")
        data = response.json()
        
        assert data["graph_loaded"] is True
        assert data["node_count"] == 1000
        assert data["edge_count"] > 0


# =============================================================================
# Tests for Agent Manifest Endpoint
# =============================================================================

@pytest.mark.api
class TestAgentManifestEndpoint:
    """Test suite for /agent.json endpoint."""
    
    def test_manifest_returns_200(self, client):
        """Test that manifest endpoint returns 200 OK."""
        response = client.get("/agent.json")
        
        assert response.status_code == 200
    
    def test_manifest_has_correct_name(self, client):
        """Test that manifest has correct agent name."""
        response = client.get("/agent.json")
        data = response.json()
        
        assert data["name"] == "green-financial-crime-agent"
    
    def test_manifest_has_version(self, client):
        """Test that manifest has version field."""
        response = client.get("/agent.json")
        data = response.json()
        
        assert "version" in data
        assert data["version"] == "1.0.0"
    
    def test_manifest_has_capabilities(self, client):
        """Test that manifest lists capabilities."""
        response = client.get("/agent.json")
        data = response.json()
        
        assert "capabilities" in data
        assert isinstance(data["capabilities"], list)
        assert len(data["capabilities"]) > 0
        
        # Check for expected capabilities
        expected = ["generate_graph", "inject_structuring", "inject_layering"]
        for cap in expected:
            assert cap in data["capabilities"]
    
    def test_manifest_has_endpoints(self, client):
        """Test that manifest lists endpoints."""
        response = client.get("/agent.json")
        data = response.json()
        
        assert "endpoints" in data
        assert isinstance(data["endpoints"], dict)
        
        # Check for expected endpoints
        expected_keys = ["health", "manifest", "transactions", "kyc"]
        for key in expected_keys:
            assert key in data["endpoints"]


# =============================================================================
# Tests for get_transactions Endpoint
# =============================================================================

@pytest.mark.api
class TestGetTransactionsEndpoint:
    """Test suite for /a2a/tools/get_transactions endpoint."""
    
    def test_returns_400_when_no_graph(self, client):
        """Test that endpoint returns 400 when no graph loaded."""
        response = client.post(
            "/a2a/tools/get_transactions",
            json={"account_id": "node_42"}
        )
        
        assert response.status_code == 400
        assert "no graph" in response.json()["detail"].lower()
    
    def test_returns_404_for_invalid_account(self, client_with_graph):
        """Test that endpoint returns 404 for non-existent account."""
        response = client_with_graph.post(
            "/a2a/tools/get_transactions",
            json={"account_id": "nonexistent_node_99999"}
        )
        
        assert response.status_code == 404
    
    def test_returns_transactions_for_valid_account(self, client_with_graph):
        """Test that endpoint returns transactions for valid account."""
        response = client_with_graph.post(
            "/a2a/tools/get_transactions",
            json={"account_id": "0"}  # Node 0 should exist in baseline_graph
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "account_id" in data
        assert "transactions" in data
        assert isinstance(data["transactions"], list)
    
    def test_respects_limit_parameter(self, client_with_graph):
        """Test that limit parameter is respected."""
        response = client_with_graph.post(
            "/a2a/tools/get_transactions",
            json={"account_id": "0", "limit": 5}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["transactions"]) <= 5
    
    def test_transaction_has_required_fields(self, client_with_graph):
        """Test that transactions have required fields."""
        response = client_with_graph.post(
            "/a2a/tools/get_transactions",
            json={"account_id": "0"}
        )
        
        data = response.json()
        
        if data["transactions"]:
            txn = data["transactions"][0]
            required_fields = ["transaction_id", "source", "target", "amount", "timestamp"]
            for field in required_fields:
                assert field in txn, f"Missing field: {field}"


# =============================================================================
# Tests for get_kyc_profile Endpoint
# =============================================================================

@pytest.mark.api
class TestGetKycProfileEndpoint:
    """Test suite for /a2a/tools/get_kyc_profile endpoint."""
    
    def test_returns_400_when_no_graph(self, client):
        """Test that endpoint returns 400 when no graph loaded."""
        response = client.post(
            "/a2a/tools/get_kyc_profile",
            json={"account_id": "node_42"}
        )
        
        assert response.status_code == 400
    
    def test_returns_404_for_invalid_account(self, client_with_graph):
        """Test that endpoint returns 404 for non-existent account."""
        response = client_with_graph.post(
            "/a2a/tools/get_kyc_profile",
            json={"account_id": "nonexistent_account"}
        )
        
        assert response.status_code == 404
    
    def test_returns_profile_for_valid_account(self, client_with_graph):
        """Test that endpoint returns profile for valid account."""
        response = client_with_graph.post(
            "/a2a/tools/get_kyc_profile",
            json={"account_id": "0"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "account_id" in data
        assert "name" in data
        assert "entity_type" in data
    
    def test_profile_has_risk_score(self, client_with_graph):
        """Test that profile includes risk score."""
        response = client_with_graph.post(
            "/a2a/tools/get_kyc_profile",
            json={"account_id": "0"}
        )
        
        data = response.json()
        
        assert "risk_score" in data
        assert 0 <= data["risk_score"] <= 1


# =============================================================================
# Tests for get_account_connections Endpoint
# =============================================================================

@pytest.mark.api
class TestGetAccountConnectionsEndpoint:
    """Test suite for /a2a/tools/get_account_connections endpoint."""
    
    def test_returns_400_when_no_graph(self, client):
        """Test that endpoint returns 400 when no graph loaded."""
        response = client.post(
            "/a2a/tools/get_account_connections",
            json={"account_id": "node_42"}
        )
        
        assert response.status_code == 400
    
    def test_returns_404_for_invalid_account(self, client_with_graph):
        """Test that endpoint returns 404 for non-existent account."""
        response = client_with_graph.post(
            "/a2a/tools/get_account_connections",
            json={"account_id": "nonexistent"}
        )
        
        assert response.status_code == 404
    
    def test_returns_connections_for_valid_account(self, client_with_graph):
        """Test that endpoint returns connections for valid account."""
        response = client_with_graph.post(
            "/a2a/tools/get_account_connections",
            json={"account_id": "0"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "account_id" in data
        assert "connections" in data
        assert "connection_count" in data


# =============================================================================
# Tests for investigation_assessment Endpoint
# =============================================================================

@pytest.mark.api
class TestInvestigationAssessmentEndpoint:
    """Test suite for /a2a/investigation_assessment endpoint."""
    
    def test_returns_400_when_no_ground_truth(self, client):
        """Test that endpoint returns 400 when no ground truth loaded."""
        response = client.post(
            "/a2a/investigation_assessment",
            json={"participant_id": "test_agent"}
        )
        
        assert response.status_code == 400
    
    def test_returns_assessment_structure(self, client_with_graph):
        """Test that endpoint returns proper assessment structure."""
        response = client_with_graph.post(
            "/a2a/investigation_assessment",
            json={"participant_id": "test_agent"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "score" in data
        assert "feedback" in data
    
    def test_assessment_score_in_range(self, client_with_graph):
        """Test that assessment score is in valid range."""
        response = client_with_graph.post(
            "/a2a/investigation_assessment",
            json={"participant_id": "test_agent"}
        )
        
        data = response.json()
        
        assert 0 <= data["score"] <= 100


# =============================================================================
# Tests for Error Handling
# =============================================================================

@pytest.mark.api
class TestErrorHandling:
    """Test suite for error handling across endpoints."""
    
    def test_invalid_json_returns_422(self, client):
        """Test that invalid JSON returns 422."""
        response = client.post(
            "/a2a/tools/get_transactions",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_missing_required_field_returns_422(self, client_with_graph):
        """Test that missing required field returns 422."""
        response = client_with_graph.post(
            "/a2a/tools/get_transactions",
            json={}  # Missing account_id
        )
        
        assert response.status_code == 422
    
    def test_invalid_method_returns_405(self, client):
        """Test that invalid HTTP method returns 405."""
        response = client.get("/a2a/tools/get_transactions")
        
        assert response.status_code == 405


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.integration
@pytest.mark.api
class TestA2AInterfaceIntegration:
    """Integration tests for A2A interface."""
    
    def test_full_workflow(self, client_with_graph):
        """Test full investigation workflow."""
        # 1. Check health
        health_resp = client_with_graph.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["graph_loaded"] is True
        
        # 2. Get manifest
        manifest_resp = client_with_graph.get("/agent.json")
        assert manifest_resp.status_code == 200
        
        # 3. Get transactions for a node
        txn_resp = client_with_graph.post(
            "/a2a/tools/get_transactions",
            json={"account_id": "0"}
        )
        assert txn_resp.status_code == 200
        
        # 4. Get KYC profile
        kyc_resp = client_with_graph.post(
            "/a2a/tools/get_kyc_profile",
            json={"account_id": "0"}
        )
        assert kyc_resp.status_code == 200
        
        # 5. Get connections
        conn_resp = client_with_graph.post(
            "/a2a/tools/get_account_connections",
            json={"account_id": "0"}
        )
        assert conn_resp.status_code == 200
        
        # 6. Request assessment
        assess_resp = client_with_graph.post(
            "/a2a/investigation_assessment",
            json={"participant_id": "integration_test"}
        )
        assert assess_resp.status_code == 200
    
    def test_manifest_endpoints_are_accessible(self, client):
        """Test that all endpoints listed in manifest are accessible."""
        # Get manifest
        manifest_resp = client.get("/agent.json")
        manifest = manifest_resp.json()
        
        # Verify health endpoint
        assert client.get(manifest["endpoints"]["health"]).status_code == 200
        
        # Verify manifest endpoint
        assert client.get(manifest["endpoints"]["manifest"]).status_code == 200
