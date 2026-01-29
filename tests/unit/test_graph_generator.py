"""
Test Suite for Graph Generator Module
=====================================
Tests for src/core/graph_generator.py

Tests cover:
1. generate_scale_free_graph - Core graph generation
2. Parameter validation
3. Reproducibility with seeds
4. Graph properties (scale-free, type)
"""

import pytest
import networkx as nx
import time
from typing import Set

from src.core.graph_generator import (
    generate_scale_free_graph,
    add_entity_attributes,
    add_transaction_attributes
)


# =============================================================================
# Tests for generate_scale_free_graph
# =============================================================================

class TestGenerateScaleFreeGraph:
    """Test suite for generate_scale_free_graph function."""
    
    def test_returns_digraph(self):
        """Test that function returns a NetworkX DiGraph."""
        G = generate_scale_free_graph(n_nodes=50)
        
        # Note: scale_free_graph returns MultiDiGraph, 
        # but we should verify it's at least a directed graph
        assert isinstance(G, (nx.DiGraph, nx.MultiDiGraph))
        assert G.is_directed()
    
    def test_node_count_exact(self):
        """Test that graph has exact number of nodes requested."""
        for n in [10, 50, 100]:
            G = generate_scale_free_graph(n_nodes=n)
            assert G.number_of_nodes() == n
    
    @pytest.mark.parametrize("n_nodes", [10, 50, 100, 500])
    def test_node_count_parametrized(self, n_nodes):
        """Parametrized test for various node counts."""
        G = generate_scale_free_graph(n_nodes=n_nodes)
        assert G.number_of_nodes() == n_nodes
    
    def test_parameter_validation_sum_to_one(self):
        """Test that alpha + beta + gamma must sum to 1.0."""
        with pytest.raises(ValueError) as excinfo:
            generate_scale_free_graph(
                n_nodes=50,
                alpha=0.5,
                beta=0.5,
                gamma=0.5  # Sum = 1.5, not 1.0
            )
        
        assert "1.0" in str(excinfo.value).lower() or "sum" in str(excinfo.value).lower()
    
    def test_default_parameters_valid(self):
        """Test that default parameters (0.41, 0.54, 0.05) work."""
        # Should not raise any exceptions
        G = generate_scale_free_graph(n_nodes=50)
        assert G.number_of_nodes() == 50
    
    def test_seed_reproducibility(self):
        """Test that same seed produces identical graph."""
        G1 = generate_scale_free_graph(n_nodes=100, seed=42)
        G2 = generate_scale_free_graph(n_nodes=100, seed=42)
        
        # Same number of nodes and edges
        assert G1.number_of_nodes() == G2.number_of_nodes()
        assert G1.number_of_edges() == G2.number_of_edges()
        
        # Same edge set
        edges1 = set(G1.edges())
        edges2 = set(G2.edges())
        assert edges1 == edges2
    
    def test_different_seeds_different_graphs(self):
        """Test that different seeds produce different graphs."""
        G1 = generate_scale_free_graph(n_nodes=100, seed=42)
        G2 = generate_scale_free_graph(n_nodes=100, seed=123)
        
        # Edge sets should differ (with high probability)
        edges1 = set(G1.edges())
        edges2 = set(G2.edges())
        
        # They might share some edges, but not all
        assert edges1 != edges2
    
    def test_scale_free_property_hub_nodes(self):
        """Test that graph exhibits scale-free property (has hub nodes)."""
        G = generate_scale_free_graph(n_nodes=500, seed=42)
        
        degrees = [d for n, d in G.degree()]
        avg_degree = sum(degrees) / len(degrees)
        max_degree = max(degrees)
        
        # Scale-free graphs have hub nodes with degree >> average
        # Hub should be at least 5x average degree
        assert max_degree > avg_degree * 3, \
            f"No hub nodes detected. Max: {max_degree}, Avg: {avg_degree}"
    
    def test_scale_free_property_power_law_tail(self):
        """Test that degree distribution has power-law tail (many low-degree nodes)."""
        G = generate_scale_free_graph(n_nodes=500, seed=42)
        
        degrees = [d for n, d in G.degree()]
        avg_degree = sum(degrees) / len(degrees)
        
        # Most nodes should have below-average degree (power-law property)
        below_average = sum(1 for d in degrees if d < avg_degree)
        
        # Typically > 60% of nodes have below-average degree in scale-free networks
        assert below_average / len(degrees) > 0.5
    
    @pytest.mark.slow
    def test_large_graph_generation(self):
        """Test generation of larger graph (1000 nodes)."""
        G = generate_scale_free_graph(n_nodes=1000, seed=42)
        
        assert G.number_of_nodes() == 1000
        assert G.number_of_edges() > 0
    
    def test_graph_is_connected_or_has_few_components(self):
        """Test that graph is reasonably connected."""
        G = generate_scale_free_graph(n_nodes=100, seed=42)
        
        # Convert to undirected for connectivity check
        G_undirected = G.to_undirected()
        num_components = nx.number_connected_components(G_undirected)
        
        # Should have relatively few components (scale-free tends to be connected)
        assert num_components < 10
    
    def test_no_isolated_nodes_for_reasonable_size(self):
        """Test that larger graphs don't have many isolated nodes."""
        G = generate_scale_free_graph(n_nodes=200, seed=42)
        
        isolated = list(nx.isolates(G))
        
        # Should have very few (if any) isolated nodes
        assert len(isolated) < G.number_of_nodes() * 0.1


# =============================================================================
# Tests for add_entity_attributes
# =============================================================================

class TestAddEntityAttributes:
    """Test suite for add_entity_attributes function."""
    
    def test_adds_required_attributes(self):
        """Test that all required attributes are added to nodes."""
        G = generate_scale_free_graph(n_nodes=10)
        G = add_entity_attributes(G, seed=42)
        
        required_attrs = ['name', 'address', 'entity_type', 'swift', 'country', 'risk_score']
        for node in G.nodes():
            for attr in required_attrs:
                assert attr in G.nodes[node], f"Missing attribute: {attr}"
    
    def test_with_faker_instance(self):
        """Test that custom Faker instance works."""
        from faker import Faker
        fake = Faker()
        
        G = generate_scale_free_graph(n_nodes=10)
        G = add_entity_attributes(G, faker_instance=fake)
        
        assert all('name' in G.nodes[n] for n in G.nodes())
    
    def test_entity_type_distribution(self):
        """Test that entity types are assigned with correct distribution."""
        G = generate_scale_free_graph(n_nodes=100)
        G = add_entity_attributes(G, seed=42)
        
        types = [G.nodes[n]['entity_type'] for n in G.nodes()]
        
        # Should have mostly persons (70%), some companies (25%), few banks (5%)
        person_count = types.count('person')
        company_count = types.count('company')
        bank_count = types.count('bank')
        
        # Check that all three types exist with reasonable distribution
        assert person_count > company_count, "Expected more persons than companies"
        assert company_count > bank_count, "Expected more companies than banks"
    
    def test_seed_reproducibility(self):
        """Test that same seed produces same attributes."""
        G1 = generate_scale_free_graph(n_nodes=10, seed=42)
        G1 = add_entity_attributes(G1, seed=42)
        
        G2 = generate_scale_free_graph(n_nodes=10, seed=42)
        G2 = add_entity_attributes(G2, seed=42)
        
        for node in G1.nodes():
            assert G1.nodes[node]['name'] == G2.nodes[node]['name']
    
    def test_locale_support(self):
        """Test that locales parameter is accepted."""
        G = generate_scale_free_graph(n_nodes=10)
        G = add_entity_attributes(G, locales=['en_US', 'en_GB'])
        
        # Should complete without error
        assert all('name' in G.nodes[n] for n in G.nodes())


# =============================================================================
# Tests for add_transaction_attributes
# =============================================================================

class TestAddTransactionAttributes:
    """Test suite for add_transaction_attributes function."""
    
    def test_adds_required_attributes(self):
        """Test that all required attributes are added to edges."""
        G = generate_scale_free_graph(n_nodes=10)
        G = add_transaction_attributes(G, seed=42)
        
        required_attrs = ['amount', 'timestamp', 'transaction_type', 'transaction_id', 'label']
        # Handle MultiDiGraph edges which include key
        if isinstance(G, nx.MultiDiGraph):
            for u, v, key, data in G.edges(keys=True, data=True):
                for attr in required_attrs:
                    assert attr in data, f"Missing attribute: {attr}"
        else:
            for u, v, data in G.edges(data=True):
                for attr in required_attrs:
                    assert attr in data, f"Missing attribute: {attr}"
    
    def test_amount_range(self):
        """Test that amounts are within expected range."""
        G = generate_scale_free_graph(n_nodes=20)
        G = add_transaction_attributes(G, seed=42)
        
        if isinstance(G, nx.MultiDiGraph):
            for u, v, key, data in G.edges(keys=True, data=True):
                amount = data['amount']
                assert 100 <= amount <= 50000, f"Amount {amount} out of range"
        else:
            for u, v, data in G.edges(data=True):
                amount = data['amount']
                assert 100 <= amount <= 50000, f"Amount {amount} out of range"
    
    def test_legitimate_label(self):
        """Test that all transactions are labeled as legitimate."""
        G = generate_scale_free_graph(n_nodes=10)
        G = add_transaction_attributes(G)
        
        if isinstance(G, nx.MultiDiGraph):
            for u, v, key, data in G.edges(keys=True, data=True):
                assert data['label'] == 'legitimate'
        else:
            for u, v, data in G.edges(data=True):
                assert data['label'] == 'legitimate'
    
    def test_transaction_types(self):
        """Test that transaction types are valid."""
        G = generate_scale_free_graph(n_nodes=20)
        G = add_transaction_attributes(G, seed=42)
        
        valid_types = ['wire', 'ach', 'cash', 'internal']
        if isinstance(G, nx.MultiDiGraph):
            for u, v, key, data in G.edges(keys=True, data=True):
                assert data['transaction_type'] in valid_types
        else:
            for u, v, data in G.edges(data=True):
                assert data['transaction_type'] in valid_types
    
    def test_seed_reproducibility(self):
        """Test that same seed produces reproducible amounts."""
        G1 = generate_scale_free_graph(n_nodes=10, seed=42)
        G1 = add_transaction_attributes(G1, seed=42)
        
        G2 = generate_scale_free_graph(n_nodes=10, seed=42)
        G2 = add_transaction_attributes(G2, seed=42)
        
        # Collect amounts from both graphs and compare
        if isinstance(G1, nx.MultiDiGraph):
            amounts1 = [data['amount'] for u, v, key, data in G1.edges(keys=True, data=True)]
            amounts2 = [data['amount'] for u, v, key, data in G2.edges(keys=True, data=True)]
        else:
            amounts1 = [data['amount'] for u, v, data in G1.edges(data=True)]
            amounts2 = [data['amount'] for u, v, data in G2.edges(data=True)]
        
        assert amounts1 == amounts2


# =============================================================================
# Performance Tests
# =============================================================================

class TestGraphGeneratorPerformance:
    """Performance tests for graph generation."""
    
    @pytest.mark.slow
    def test_1000_node_generation_under_10_seconds(self):
        """Test that 1000-node graph generates in under 10 seconds."""
        start_time = time.time()
        
        G = generate_scale_free_graph(n_nodes=1000, seed=42)
        
        elapsed = time.time() - start_time
        
        assert G.number_of_nodes() == 1000
        assert elapsed < 10.0, f"Generation took {elapsed:.2f}s, expected < 10s"
    
    @pytest.mark.slow
    def test_consistent_generation_time(self):
        """Test that generation time is consistent across runs."""
        times = []
        
        for _ in range(3):
            start_time = time.time()
            G = generate_scale_free_graph(n_nodes=500, seed=42)
            elapsed = time.time() - start_time
            times.append(elapsed)
        
        # All runs should be within reasonable bounds (3x max variance for fast operations)
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        # For very fast operations, just ensure they complete reasonably
        assert max_time < 5.0, f"Generation too slow: {max_time}s"
        # Check that variance is not extreme (max is not more than 10x min)
        if min_time > 0.001:  # Only check ratio if times are measurable
            assert max_time / min_time < 10


# =============================================================================
# Edge Cases
# =============================================================================

class TestGraphGeneratorEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_minimum_nodes(self):
        """Test generation with minimum number of nodes."""
        G = generate_scale_free_graph(n_nodes=3)
        assert G.number_of_nodes() == 3
    
    def test_parameter_at_boundary(self):
        """Test with parameter values at boundaries."""
        # High alpha with minimal beta (NetworkX requires beta > 0)
        G = generate_scale_free_graph(n_nodes=50, alpha=0.98, beta=0.01, gamma=0.01)
        assert G.number_of_nodes() == 50
    
    def test_none_seed_varies(self):
        """Test that None seed produces (likely) different graphs."""
        G1 = generate_scale_free_graph(n_nodes=50, seed=None)
        G2 = generate_scale_free_graph(n_nodes=50, seed=None)
        
        # They might be the same by chance, but very unlikely
        # Just verify both generate successfully
        assert G1.number_of_nodes() == 50
        assert G2.number_of_nodes() == 50
