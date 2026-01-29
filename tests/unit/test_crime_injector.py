"""
Test Suite for Crime Injector Module
====================================
Tests for src/core/crime_injector.py

Test Categories:
1. Config dataclasses (StructuringConfig, LayeringConfig)
2. validate_no_cycles function (IMPLEMENTED)
3. inject_structuring function (NOT YET IMPLEMENTED - skipped)
4. inject_layering function (NOT YET IMPLEMENTED - skipped)
5. get_crime_labels function (NOT YET IMPLEMENTED - skipped)
"""

import pytest
import networkx as nx
from datetime import datetime, timedelta

from src.core.crime_injector import (
    StructuringConfig,
    LayeringConfig,
    InjectedCrime,
    inject_structuring,
    inject_layering,
    validate_no_cycles,
    get_crime_labels
)


# =============================================================================
# Tests for Config Dataclasses
# =============================================================================

class TestStructuringConfig:
    """Test suite for StructuringConfig dataclass."""
    
    def test_default_values(self):
        """Test that default values are correct."""
        config = StructuringConfig()
        
        assert config.num_sources == 20
        assert config.mule_node is None
        assert config.min_amount == 9000.0
        assert config.max_amount == 9800.0
        assert config.time_window_hours == 48
    
    def test_custom_values(self):
        """Test that custom values can be set."""
        config = StructuringConfig(
            num_sources=15,
            mule_node=42,
            min_amount=8500.0,
            max_amount=9500.0,
            time_window_hours=24
        )
        
        assert config.num_sources == 15
        assert config.mule_node == 42
        assert config.min_amount == 8500.0
        assert config.max_amount == 9500.0
        assert config.time_window_hours == 24
    
    def test_amount_range_valid(self):
        """Test that default amount range is below CTR threshold."""
        config = StructuringConfig()
        
        assert config.max_amount < 10000  # CTR threshold
        assert config.min_amount < config.max_amount


class TestLayeringConfig:
    """Test suite for LayeringConfig dataclass."""
    
    def test_default_values(self):
        """Test that default values are correct."""
        config = LayeringConfig()
        
        assert config.chain_length == 5
        assert config.min_decay == 0.02
        assert config.max_decay == 0.05
        assert config.initial_amount == 100000.0
    
    def test_custom_values(self):
        """Test that custom values can be set."""
        config = LayeringConfig(
            chain_length=7,
            min_decay=0.03,
            max_decay=0.08,
            initial_amount=50000.0
        )
        
        assert config.chain_length == 7
        assert config.min_decay == 0.03
        assert config.max_decay == 0.08
        assert config.initial_amount == 50000.0
    
    def test_decay_range_valid(self):
        """Test that decay range is reasonable (2-5%)."""
        config = LayeringConfig()
        
        assert 0 < config.min_decay < config.max_decay < 1
        assert config.min_decay == 0.02  # 2%
        assert config.max_decay == 0.05  # 5%


class TestInjectedCrime:
    """Test suite for InjectedCrime dataclass."""
    
    def test_creation(self):
        """Test that InjectedCrime can be created."""
        crime = InjectedCrime(
            crime_type="structuring",
            nodes_involved=[1, 2, 3],
            edges_involved=[(1, 2), (2, 3)],
            metadata={"key": "value"}
        )
        
        assert crime.crime_type == "structuring"
        assert len(crime.nodes_involved) == 3
        assert len(crime.edges_involved) == 2
        assert crime.metadata["key"] == "value"


# =============================================================================
# Tests for validate_no_cycles (IMPLEMENTED)
# =============================================================================

class TestValidateNoCycles:
    """Test suite for validate_no_cycles function."""
    
    def test_simple_chain_no_cycles(self):
        """Test that simple chain has no cycles."""
        G = nx.DiGraph()
        edges = [(1, 2), (2, 3), (3, 4), (4, 5)]
        G.add_edges_from(edges)
        
        result = validate_no_cycles(G, edges)
        
        assert result is True
    
    def test_cycle_detected(self):
        """Test that cycle is detected."""
        G = nx.DiGraph()
        edges = [(1, 2), (2, 3), (3, 1)]  # Cycle: 1->2->3->1
        G.add_edges_from(edges)
        
        result = validate_no_cycles(G, edges)
        
        assert result is False
    
    def test_self_loop_is_cycle(self):
        """Test that self-loop is detected as cycle."""
        G = nx.DiGraph()
        edges = [(1, 2), (2, 2), (2, 3)]  # Self-loop at 2
        G.add_edges_from(edges)
        
        result = validate_no_cycles(G, edges)
        
        assert result is False
    
    def test_single_edge_no_cycle(self):
        """Test that single edge has no cycle."""
        G = nx.DiGraph()
        edges = [(1, 2)]
        G.add_edges_from(edges)
        
        result = validate_no_cycles(G, edges)
        
        assert result is True
    
    def test_empty_edges(self):
        """Test with empty edge list."""
        G = nx.DiGraph()
        
        # Empty edge list should not have cycles
        result = validate_no_cycles(G, [])
        
        # Behavior depends on implementation
        # With edge_subgraph on empty, should be True (no cycles)
        assert result is True
    
    def test_long_chain_no_cycle(self):
        """Test that long chain without back-edge has no cycle."""
        G = nx.DiGraph()
        chain_length = 20
        edges = [(i, i + 1) for i in range(chain_length)]
        G.add_edges_from(edges)
        
        result = validate_no_cycles(G, edges)
        
        assert result is True
    
    def test_cycle_at_end_of_chain(self):
        """Test that cycle at end of chain is detected."""
        G = nx.DiGraph()
        # Chain 1->2->3->4->5 with cycle 5->3
        edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 3)]
        G.add_edges_from(edges)
        
        result = validate_no_cycles(G, edges)
        
        assert result is False
    
    def test_partial_edge_list(self):
        """Test with partial edge list (subset of graph)."""
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])  # Full graph has cycle
        
        # But check only the chain portion
        chain_edges = [(1, 2), (2, 3), (3, 4)]
        
        result = validate_no_cycles(G, chain_edges)
        
        assert result is True  # The subset doesn't have a cycle


# =============================================================================
# Tests for inject_structuring
# =============================================================================

class TestInjectStructuring:
    """Test suite for inject_structuring function."""
    
    def test_creates_correct_number_of_sources(self, small_graph):
        """Test structuring creates correct number of source nodes."""
        config = StructuringConfig(num_sources=20)
        original_count = small_graph.number_of_nodes()
        
        G, crime = inject_structuring(small_graph, config, seed=42)
        
        assert G.number_of_nodes() == original_count + config.num_sources
    
    def test_all_amounts_below_ctr(self, small_graph):
        """Test all structuring amounts are below $10,000 CTR threshold."""
        config = StructuringConfig()
        
        G, crime = inject_structuring(small_graph, config, seed=42)
        
        for edge in crime.edges_involved:
            amount = G.edges[edge].get('amount', 0)
            assert amount < 10000
            assert config.min_amount <= amount <= config.max_amount
    
    def test_single_target_mule(self, small_graph):
        """Test all transfers go to single mule node."""
        config = StructuringConfig()
        
        G, crime = inject_structuring(small_graph, config, seed=42)
        
        targets = set(edge[1] for edge in crime.edges_involved)
        assert len(targets) == 1
    
    def test_edges_labeled_correctly(self, small_graph):
        """Test crime edges are labeled as 'structuring'."""
        config = StructuringConfig()
        
        G, crime = inject_structuring(small_graph, config, seed=42)
        
        for edge in crime.edges_involved:
            assert G.edges[edge].get('label') == 'structuring'
    
    def test_returns_injected_crime_record(self, small_graph):
        """Test that InjectedCrime record is returned."""
        config = StructuringConfig()
        
        G, crime = inject_structuring(small_graph, config, seed=42)
        
        assert isinstance(crime, InjectedCrime)
        assert crime.crime_type == 'structuring'
    
    def test_metadata_contains_mule_id(self, small_graph):
        """Test that metadata contains mule node ID."""
        G, crime = inject_structuring(small_graph, seed=42)
        
        assert 'mule_id' in crime.metadata
        assert crime.metadata['mule_id'] in G.nodes()
    
    def test_custom_mule_node(self, small_graph):
        """Test that custom mule node can be specified."""
        mule_id = list(small_graph.nodes())[0]
        config = StructuringConfig(mule_node=mule_id)
        
        G, crime = inject_structuring(small_graph, config, seed=42)
        
        assert crime.metadata['mule_id'] == mule_id
        for edge in crime.edges_involved:
            assert edge[1] == mule_id


# =============================================================================
# Tests for inject_layering
# =============================================================================

class TestInjectLayering:
    """Test suite for inject_layering function."""
    
    def test_creates_chain_of_correct_length(self, small_graph):
        """Test layering creates chain of correct length."""
        config = LayeringConfig(chain_length=6)
        
        G, crime = inject_layering(small_graph, config, seed=42)
        
        # Chain length of 6 means 6 intermediate nodes + source + dest = 8 nodes in chain
        # Which gives 7 edges (chain_length + 1)
        assert len(crime.edges_involved) == config.chain_length + 1
    
    def test_no_cycles_in_chain(self, small_graph):
        """Test that injected chain has no cycles."""
        config = LayeringConfig()
        
        G, crime = inject_layering(small_graph, config, seed=42)
        
        assert validate_no_cycles(G, crime.edges_involved) is True
    
    def test_amounts_decay(self, small_graph):
        """Test that amounts decay along the chain."""
        config = LayeringConfig()
        
        G, crime = inject_layering(small_graph, config, seed=42)
        
        amounts = [G.edges[e]['amount'] for e in crime.edges_involved]
        
        # Each amount should be less than or equal to the previous (decay starts after first hop)
        for i in range(2, len(amounts)):
            assert amounts[i] < amounts[i - 1], f"Amount did not decay at hop {i}"
    
    def test_decay_rate_in_range(self, small_graph):
        """Test that decay rate is within 2-5% per hop."""
        config = LayeringConfig(min_decay=0.02, max_decay=0.05)
        
        G, crime = inject_layering(small_graph, config, seed=42)
        
        amounts = [G.edges[e]['amount'] for e in crime.edges_involved]
        
        # Check decay rates starting from second edge
        for i in range(2, len(amounts)):
            decay = 1 - (amounts[i] / amounts[i - 1])
            assert config.min_decay <= decay <= config.max_decay, \
                f"Decay {decay} at hop {i} not in range [{config.min_decay}, {config.max_decay}]"
    
    def test_edges_labeled_correctly(self, small_graph):
        """Test crime edges are labeled as 'layering'."""
        config = LayeringConfig()
        
        G, crime = inject_layering(small_graph, config, seed=42)
        
        for edge in crime.edges_involved:
            assert G.edges[edge].get('label') == 'layering'
    
    def test_returns_injected_crime_record(self, small_graph):
        """Test that InjectedCrime record is returned."""
        config = LayeringConfig()
        
        G, crime = inject_layering(small_graph, config, seed=42)
        
        assert isinstance(crime, InjectedCrime)
        assert crime.crime_type == 'layering'
    
    def test_metadata_contains_chain_info(self, small_graph):
        """Test that metadata contains chain information."""
        G, crime = inject_layering(small_graph, seed=42)
        
        assert 'chain_length' in crime.metadata
        assert 'initial_amount' in crime.metadata
        assert 'final_amount' in crime.metadata
        assert crime.metadata['final_amount'] < crime.metadata['initial_amount']


# =============================================================================
# Tests for get_crime_labels
# =============================================================================

class TestGetCrimeLabels:
    """Test suite for get_crime_labels function."""
    
    def test_returns_dict(self, small_graph):
        """Test that function returns a dictionary."""
        labels = get_crime_labels(small_graph)
        
        assert isinstance(labels, dict)
    
    def test_all_edges_have_labels(self, small_graph):
        """Test that all edges have labels in result."""
        labels = get_crime_labels(small_graph)
        
        for edge in small_graph.edges():
            assert edge in labels
    
    def test_legitimate_labels_default(self, small_graph):
        """Test that edges without label attribute return 'legitimate'."""
        # Remove label attribute from edges
        for u, v in small_graph.edges():
            if 'label' in small_graph.edges[u, v]:
                del small_graph.edges[u, v]['label']
        
        labels = get_crime_labels(small_graph)
        
        for label in labels.values():
            assert label == 'legitimate'
    
    def test_crime_labels_preserved(self, small_graph):
        """Test that crime labels are correctly extracted."""
        # Inject a structuring crime
        G, crime = inject_structuring(small_graph, seed=42)
        
        labels = get_crime_labels(G)
        
        # Check that crime edges are labeled correctly
        for edge in crime.edges_involved:
            assert labels[edge] == 'structuring'
    
    def test_mixed_labels(self, small_graph):
        """Test extraction of mixed crime and legitimate labels."""
        # Inject both crime types
        G, structuring_crime = inject_structuring(small_graph, seed=42)
        G, layering_crime = inject_layering(G, seed=43)
        
        labels = get_crime_labels(G)
        
        # Verify structuring edges
        for edge in structuring_crime.edges_involved:
            assert labels[edge] == 'structuring'
        
        # Verify layering edges
        for edge in layering_crime.edges_involved:
            assert labels[edge] == 'layering'


# =============================================================================
# Integration Tests
# =============================================================================

class TestCrimeInjectorIntegration:
    """Integration tests for crime injector module."""
    
    def test_validate_no_cycles_with_layering_pattern(self, valid_layering_chain):
        """Test validate_no_cycles with valid layering chain fixture."""
        G, chain_edges = valid_layering_chain
        
        result = validate_no_cycles(G, chain_edges)
        
        assert result is True
    
    def test_validate_no_cycles_with_cycle_pattern(self, invalid_layering_chain_with_cycle):
        """Test validate_no_cycles with invalid chain containing cycle."""
        G, chain_edges = invalid_layering_chain_with_cycle
        
        result = validate_no_cycles(G, chain_edges)
        
        assert result is False
    
    def test_config_classes_are_dataclasses(self):
        """Test that config classes are properly defined dataclasses."""
        from dataclasses import is_dataclass
        
        assert is_dataclass(StructuringConfig)
        assert is_dataclass(LayeringConfig)
        assert is_dataclass(InjectedCrime)
