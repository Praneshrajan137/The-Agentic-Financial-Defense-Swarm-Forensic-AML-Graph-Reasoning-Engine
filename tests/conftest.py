"""
Pytest Configuration and Shared Fixtures
========================================
Provides reusable fixtures for the Green Financial Crime Agent test suite.

Fixtures:
- small_graph: 50-node graph for fast unit tests
- baseline_graph: 1000-node graph for integration tests
- faker_instance: Seeded Faker for reproducible entity generation
- test_client: FastAPI TestClient for API testing
- sample_edges_with_attrs: Sample edge data for validator tests
"""

import pytest
import networkx as nx
from faker import Faker
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta
import random
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Graph Fixtures
# =============================================================================

@pytest.fixture
def small_graph() -> nx.DiGraph:
    """
    Fixture providing small test graph (50 nodes) for fast unit tests.
    
    Returns:
        NetworkX DiGraph with basic node attributes
    """
    G = nx.scale_free_graph(50, alpha=0.41, beta=0.54, gamma=0.05, seed=42)
    G = nx.DiGraph(G)  # Convert MultiDiGraph to DiGraph
    
    fake = Faker()
    Faker.seed(42)
    random.seed(42)
    
    for node in G.nodes():
        G.nodes[node]['id'] = f"node_{node}"
        G.nodes[node]['name'] = fake.name()
        G.nodes[node]['address'] = fake.address().replace('\n', ', ')
        G.nodes[node]['entity_type'] = random.choice(['person', 'company', 'bank'])
        G.nodes[node]['risk_score'] = round(random.uniform(0, 1), 2)
    
    # Add edge attributes
    for u, v in G.edges():
        G.edges[u, v]['amount'] = round(random.uniform(100, 10000), 2)
        G.edges[u, v]['timestamp'] = datetime.now().isoformat()
        G.edges[u, v]['label'] = 'legitimate'
    
    return G


@pytest.fixture
def baseline_graph() -> nx.DiGraph:
    """
    Fixture providing standard 1000-node graph for integration tests.
    
    Returns:
        NetworkX DiGraph with full node and edge attributes
    """
    G = nx.scale_free_graph(1000, alpha=0.41, beta=0.54, gamma=0.05, seed=42)
    G = nx.DiGraph(G)  # Convert MultiDiGraph to DiGraph
    
    fake = Faker(['en_US', 'en_GB', 'en_IN'])
    Faker.seed(42)
    random.seed(42)
    
    entity_types = ['person', 'company', 'bank']
    entity_weights = [0.7, 0.25, 0.05]
    
    for node in G.nodes():
        entity_type = random.choices(entity_types, weights=entity_weights)[0]
        G.nodes[node]['id'] = f"node_{node}"
        G.nodes[node]['entity_type'] = entity_type
        G.nodes[node]['name'] = fake.name() if entity_type == 'person' else fake.company()
        G.nodes[node]['address'] = fake.address().replace('\n', ', ')
        G.nodes[node]['swift'] = fake.swift()
        G.nodes[node]['country'] = fake.country_code()
        G.nodes[node]['risk_score'] = round(random.uniform(0, 1), 2)
        G.nodes[node]['verification_status'] = 'verified'
    
    base_time = datetime.now()
    for u, v in G.edges():
        G.edges[u, v]['transaction_id'] = f"txn_{random.randint(10000, 99999)}"
        G.edges[u, v]['amount'] = round(random.uniform(100, 50000), 2)
        G.edges[u, v]['currency'] = 'USD'
        G.edges[u, v]['timestamp'] = (base_time - timedelta(days=random.randint(0, 365))).isoformat()
        G.edges[u, v]['transaction_type'] = random.choice(['wire', 'ach', 'cash'])
        G.edges[u, v]['label'] = 'legitimate'
    
    return G


@pytest.fixture
def empty_graph() -> nx.DiGraph:
    """Fixture providing empty graph for edge case testing."""
    return nx.DiGraph()


@pytest.fixture
def graph_with_self_loops() -> nx.DiGraph:
    """Fixture providing graph with self-loops for validation testing."""
    G = nx.DiGraph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])
    G.add_edge(2, 2)  # Self-loop
    G.add_edge(4, 4)  # Self-loop
    return G


@pytest.fixture
def graph_with_isolated_nodes() -> nx.DiGraph:
    """Fixture providing graph with isolated nodes for validation testing."""
    G = nx.DiGraph()
    G.add_nodes_from([1, 2, 3, 4, 5, 6, 7])  # 7 nodes
    G.add_edges_from([(1, 2), (2, 3), (3, 4)])  # Only 4 connected
    # Nodes 5, 6, 7 are isolated
    return G


# =============================================================================
# Faker Fixtures
# =============================================================================

@pytest.fixture
def faker_instance() -> Faker:
    """
    Fixture providing seeded Faker instance for reproducible tests.
    
    Returns:
        Faker instance with multiple locales, seeded for reproducibility
    """
    fake = Faker(['en_US', 'en_GB', 'en_IN'])
    Faker.seed(42)
    return fake


# =============================================================================
# API Testing Fixtures
# =============================================================================

@pytest.fixture
def test_client():
    """
    Fixture providing FastAPI TestClient for API testing.
    
    Returns:
        TestClient instance for the A2A interface
    """
    from fastapi.testclient import TestClient
    from src.core.a2a_interface import app
    
    return TestClient(app)


@pytest.fixture
def test_client_with_graph(baseline_graph):
    """
    Fixture providing TestClient with a graph already loaded.
    
    Returns:
        TestClient with graph state set
    """
    from fastapi.testclient import TestClient
    from src.core.a2a_interface import app, set_graph
    
    set_graph(baseline_graph)
    client = TestClient(app)
    yield client
    # Cleanup: reset graph
    set_graph(None)


# =============================================================================
# Sample Data Fixtures for Validators
# =============================================================================

@pytest.fixture
def valid_structuring_edges() -> List[Tuple[int, int, Dict[str, Any]]]:
    """
    Fixture providing valid structuring pattern edges.
    
    Returns:
        List of edges with valid structuring attributes (single target, amounts <$10k)
    """
    mule_id = 100
    edges = []
    base_time = datetime.now()
    
    for i in range(20):
        source_id = 200 + i
        amount = round(random.uniform(9000, 9800), 2)
        timestamp = base_time + timedelta(hours=random.randint(0, 48))
        
        edges.append((
            source_id,
            mule_id,
            {
                'amount': amount,
                'timestamp': timestamp.isoformat(),
                'label': 'structuring'
            }
        ))
    
    return edges


@pytest.fixture
def invalid_structuring_edges() -> List[Tuple[int, int, Dict[str, Any]]]:
    """
    Fixture providing invalid structuring pattern edges (multiple targets).
    
    Returns:
        List of edges targeting multiple mules (invalid pattern)
    """
    edges = [
        (1, 100, {'amount': 9500, 'label': 'structuring'}),  # Target 100
        (2, 100, {'amount': 9200, 'label': 'structuring'}),
        (3, 200, {'amount': 9700, 'label': 'structuring'}),  # Target 200 - invalid!
        (4, 100, {'amount': 9100, 'label': 'structuring'}),
    ]
    return edges


@pytest.fixture
def valid_layering_chain() -> Tuple[nx.DiGraph, List[Tuple[int, int]]]:
    """
    Fixture providing valid layering chain (no cycles).
    
    Returns:
        Tuple of (graph, chain_edges)
    """
    G = nx.DiGraph()
    chain_nodes = [1, 2, 3, 4, 5, 6]
    
    for node in chain_nodes:
        G.add_node(node)
    
    chain_edges = []
    for i in range(len(chain_nodes) - 1):
        G.add_edge(chain_nodes[i], chain_nodes[i + 1])
        chain_edges.append((chain_nodes[i], chain_nodes[i + 1]))
    
    return G, chain_edges


@pytest.fixture
def invalid_layering_chain_with_cycle() -> Tuple[nx.DiGraph, List[Tuple[int, int]]]:
    """
    Fixture providing invalid layering chain (contains cycle).
    
    Returns:
        Tuple of (graph, chain_edges) where chain has a cycle
    """
    G = nx.DiGraph()
    G.add_nodes_from([1, 2, 3, 4, 5])
    
    # Create chain with cycle: 1->2->3->4->5->2 (cycle back to 2)
    chain_edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 2)]
    for u, v in chain_edges:
        G.add_edge(u, v)
    
    return G, chain_edges


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests"
    )
    config.addinivalue_line(
        "markers", "pending: marks tests for not-yet-implemented features"
    )


# =============================================================================
# Helper Functions
# =============================================================================

def create_chain_graph(length: int) -> Tuple[nx.DiGraph, List[Tuple[int, int]]]:
    """
    Helper to create a simple chain graph.
    
    Args:
        length: Number of edges in the chain
    
    Returns:
        Tuple of (graph, edge_list)
    """
    G = nx.DiGraph()
    nodes = list(range(length + 1))
    G.add_nodes_from(nodes)
    
    edges = [(i, i + 1) for i in range(length)]
    G.add_edges_from(edges)
    
    return G, edges


def create_graph_with_crime_labels(
    n_nodes: int,
    structuring_count: int = 0,
    layering_count: int = 0
) -> nx.DiGraph:
    """
    Helper to create graph with specific crime patterns for testing.
    
    Args:
        n_nodes: Base number of nodes
        structuring_count: Number of structuring edges to add
        layering_count: Number of layering edges to add
    
    Returns:
        Graph with labeled crime edges
    """
    G = nx.scale_free_graph(n_nodes, alpha=0.41, beta=0.54, gamma=0.05, seed=42)
    G = nx.DiGraph(G)
    
    # Label all edges as legitimate
    for u, v in G.edges():
        G.edges[u, v]['label'] = 'legitimate'
        G.edges[u, v]['amount'] = round(random.uniform(100, 50000), 2)
    
    # Add structuring edges
    if structuring_count > 0:
        mule = max(G.nodes()) + 1
        G.add_node(mule)
        for i in range(structuring_count):
            source = mule + 1 + i
            G.add_node(source)
            G.add_edge(source, mule, label='structuring', amount=round(random.uniform(9000, 9800), 2))
    
    return G
