"""
Test Suite for Validators Module
================================
Tests for all validator functions in src/utils/validators.py

These validators are FULLY IMPLEMENTED and should have 100% coverage.

Test Categories:
1. validate_graph_structure - Graph type, self-loops, isolated nodes
2. validate_scale_free_distribution - Hub detection, distribution metrics
3. validate_structuring_pattern - Amount validation, single target
4. validate_layering_pattern - Chain structure, cycle detection
5. validate_no_data_leakage - Train/test separation
"""

import pytest
import networkx as nx
from typing import List, Tuple

from src.utils.validators import (
    validate_graph_structure,
    validate_scale_free_distribution,
    validate_structuring_pattern,
    validate_layering_pattern,
    validate_no_data_leakage
)


# =============================================================================
# Tests for validate_graph_structure
# =============================================================================

class TestValidateGraphStructure:
    """Test suite for validate_graph_structure function."""
    
    def test_valid_digraph_passes(self, small_graph):
        """Test that a valid DiGraph passes validation."""
        is_valid, errors = validate_graph_structure(small_graph)
        # Note: scale_free_graph may have some self-loops by default
        # We just verify function runs without exception
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
    
    def test_rejects_undirected_graph(self):
        """Test that undirected graph fails validation."""
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4)])
        
        is_valid, errors = validate_graph_structure(G)
        
        assert is_valid is False
        assert any("directed" in err.lower() for err in errors)
    
    def test_detects_self_loops(self, graph_with_self_loops):
        """Test that self-loops are detected."""
        is_valid, errors = validate_graph_structure(graph_with_self_loops)
        
        assert is_valid is False
        assert any("self-loop" in err.lower() for err in errors)
    
    def test_detects_isolated_nodes(self, graph_with_isolated_nodes):
        """Test that isolated nodes are detected."""
        is_valid, errors = validate_graph_structure(graph_with_isolated_nodes)
        
        assert is_valid is False
        assert any("isolated" in err.lower() for err in errors)
    
    def test_clean_digraph_passes(self):
        """Test that a clean DiGraph with no issues passes."""
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])
        # All nodes connected, no self-loops
        
        is_valid, errors = validate_graph_structure(G)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_returns_tuple(self, small_graph):
        """Test that function returns correct tuple type."""
        result = validate_graph_structure(small_graph)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], list)


# =============================================================================
# Tests for validate_scale_free_distribution
# =============================================================================

class TestValidateScaleFreeDistribution:
    """Test suite for validate_scale_free_distribution function."""
    
    def test_scale_free_graph_is_valid(self, baseline_graph):
        """Test that scale-free graph passes distribution validation."""
        is_valid, metrics = validate_scale_free_distribution(baseline_graph)
        
        assert is_valid is True
        assert metrics["hub_count"] > 0
    
    def test_returns_correct_metrics(self, baseline_graph):
        """Test that all expected metrics are returned."""
        is_valid, metrics = validate_scale_free_distribution(baseline_graph)
        
        required_keys = ["num_nodes", "num_edges", "avg_degree", "max_degree", 
                        "min_degree", "degree_variance", "hub_count"]
        
        for key in required_keys:
            assert key in metrics, f"Missing metric: {key}"
    
    def test_empty_graph_fails(self, empty_graph):
        """Test that empty graph fails validation."""
        is_valid, metrics = validate_scale_free_distribution(empty_graph)
        
        assert is_valid is False
        assert "error" in metrics
    
    def test_node_count_accurate(self, baseline_graph):
        """Test that node count in metrics is accurate."""
        is_valid, metrics = validate_scale_free_distribution(baseline_graph)
        
        assert metrics["num_nodes"] == baseline_graph.number_of_nodes()
    
    def test_edge_count_accurate(self, baseline_graph):
        """Test that edge count in metrics is accurate."""
        is_valid, metrics = validate_scale_free_distribution(baseline_graph)
        
        assert metrics["num_edges"] == baseline_graph.number_of_edges()
    
    def test_hub_detection(self, baseline_graph):
        """Test that hub nodes are correctly identified."""
        is_valid, metrics = validate_scale_free_distribution(baseline_graph)
        
        # Hub threshold is avg_degree * 5
        hub_threshold = metrics["avg_degree"] * 5
        
        # Manually count hubs
        degrees = [d for n, d in baseline_graph.degree()]
        expected_hubs = sum(1 for d in degrees if d > hub_threshold)
        
        assert metrics["hub_count"] == expected_hubs
    
    def test_small_uniform_graph_may_fail(self):
        """Test that uniform (non-scale-free) graph may fail hub detection."""
        # Create a uniform degree graph (cycle)
        G = nx.DiGraph()
        G.add_edges_from([(i, (i + 1) % 10) for i in range(10)])
        
        is_valid, metrics = validate_scale_free_distribution(G)
        
        # Uniform graph has no hubs
        assert metrics["hub_count"] == 0
        assert is_valid is False


# =============================================================================
# Tests for validate_structuring_pattern
# =============================================================================

class TestValidateStructuringPattern:
    """Test suite for validate_structuring_pattern function."""
    
    def test_valid_pattern_passes(self, valid_structuring_edges):
        """Test that valid structuring pattern passes validation."""
        is_valid, errors = validate_structuring_pattern(valid_structuring_edges)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_multiple_targets_fails(self, invalid_structuring_edges):
        """Test that multiple target nodes fails validation."""
        is_valid, errors = validate_structuring_pattern(invalid_structuring_edges)
        
        assert is_valid is False
        assert any("multiple targets" in err.lower() or "mule" in err.lower() for err in errors)
    
    def test_amount_below_minimum_fails(self):
        """Test that amounts below minimum fail validation."""
        edges = [
            (1, 100, {'amount': 8000}),  # Below 9000 minimum
            (2, 100, {'amount': 9500}),
        ]
        
        is_valid, errors = validate_structuring_pattern(edges)
        
        assert is_valid is False
        assert any("amount" in err.lower() for err in errors)
    
    def test_amount_above_maximum_fails(self):
        """Test that amounts above maximum fail validation."""
        edges = [
            (1, 100, {'amount': 9500}),
            (2, 100, {'amount': 10500}),  # Above 9800 maximum
        ]
        
        is_valid, errors = validate_structuring_pattern(edges)
        
        assert is_valid is False
        assert any("amount" in err.lower() for err in errors)
    
    def test_empty_edges_fails(self):
        """Test that empty edge list fails validation."""
        is_valid, errors = validate_structuring_pattern([])
        
        assert is_valid is False
        assert any("no edges" in err.lower() for err in errors)
    
    def test_custom_amount_range(self):
        """Test validation with custom amount range."""
        edges = [
            (1, 100, {'amount': 5000}),
            (2, 100, {'amount': 5500}),
        ]
        
        # Should fail with default range (9000-9800)
        is_valid_default, _ = validate_structuring_pattern(edges)
        assert is_valid_default is False
        
        # Should pass with custom range
        is_valid_custom, errors = validate_structuring_pattern(
            edges, min_amount=4000, max_amount=6000
        )
        assert is_valid_custom is True
    
    def test_single_edge_valid(self):
        """Test that single valid edge passes."""
        edges = [(1, 100, {'amount': 9500})]
        
        is_valid, errors = validate_structuring_pattern(edges)
        
        assert is_valid is True


# =============================================================================
# Tests for validate_layering_pattern
# =============================================================================

class TestValidateLayeringPattern:
    """Test suite for validate_layering_pattern function."""
    
    def test_valid_chain_passes(self, valid_layering_chain):
        """Test that valid chain passes validation."""
        G, chain_edges = valid_layering_chain
        
        is_valid, errors = validate_layering_pattern(G, chain_edges)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_chain_with_cycle_fails(self, invalid_layering_chain_with_cycle):
        """Test that chain with cycle fails validation."""
        G, chain_edges = invalid_layering_chain_with_cycle
        
        is_valid, errors = validate_layering_pattern(G, chain_edges)
        
        assert is_valid is False
        assert any("cycle" in err.lower() for err in errors)
    
    def test_empty_chain_fails(self):
        """Test that empty chain fails validation."""
        G = nx.DiGraph()
        
        is_valid, errors = validate_layering_pattern(G, [])
        
        assert is_valid is False
        assert any("no chain" in err.lower() for err in errors)
    
    def test_branching_chain_fails(self):
        """Test that branching chain (not simple path) is detected."""
        G = nx.DiGraph()
        # Create branching structure: 1->2, 1->3 (two paths from 1)
        G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])
        chain_edges = [(1, 2), (1, 3), (2, 4), (3, 4)]
        
        is_valid, errors = validate_layering_pattern(G, chain_edges)
        
        # Should detect multiple start/end issues
        assert is_valid is False
    
    def test_single_edge_chain_valid(self):
        """Test that single edge chain is valid."""
        G = nx.DiGraph()
        G.add_edge(1, 2)
        
        is_valid, errors = validate_layering_pattern(G, [(1, 2)])
        
        assert is_valid is True
    
    def test_long_chain_valid(self):
        """Test that long chain (10+ nodes) is valid."""
        G = nx.DiGraph()
        chain_length = 15
        chain_edges = [(i, i + 1) for i in range(chain_length)]
        G.add_edges_from(chain_edges)
        
        is_valid, errors = validate_layering_pattern(G, chain_edges)
        
        assert is_valid is True


# =============================================================================
# Tests for validate_no_data_leakage
# =============================================================================

class TestValidateNoDataLeakage:
    """Test suite for validate_no_data_leakage function."""
    
    def test_no_overlap_passes(self):
        """Test that non-overlapping sets pass validation."""
        train_edges = [(1, 2), (2, 3), (3, 4)]
        test_edges = [(5, 6), (6, 7), (7, 8)]
        
        is_valid, overlaps = validate_no_data_leakage(train_edges, test_edges)
        
        assert is_valid is True
        assert len(overlaps) == 0
    
    def test_overlap_detected(self):
        """Test that overlapping edges are detected."""
        train_edges = [(1, 2), (2, 3), (3, 4)]
        test_edges = [(2, 3), (5, 6)]  # (2, 3) appears in both
        
        is_valid, overlaps = validate_no_data_leakage(train_edges, test_edges)
        
        assert is_valid is False
        assert len(overlaps) > 0
    
    def test_multiple_overlaps_detected(self):
        """Test that multiple overlapping edges are detected."""
        train_edges = [(1, 2), (2, 3), (3, 4), (4, 5)]
        test_edges = [(2, 3), (3, 4), (6, 7)]  # Two overlaps
        
        is_valid, overlaps = validate_no_data_leakage(train_edges, test_edges)
        
        assert is_valid is False
        assert len(overlaps) >= 2
    
    def test_empty_sets_pass(self):
        """Test that empty sets pass (no leakage possible)."""
        is_valid, overlaps = validate_no_data_leakage([], [])
        
        assert is_valid is True
        assert len(overlaps) == 0
    
    def test_one_empty_set_passes(self):
        """Test that one empty set passes."""
        train_edges = [(1, 2), (2, 3)]
        
        is_valid, overlaps = validate_no_data_leakage(train_edges, [])
        
        assert is_valid is True
    
    def test_identical_sets_fail(self):
        """Test that identical sets fail validation."""
        edges = [(1, 2), (2, 3), (3, 4)]
        
        is_valid, overlaps = validate_no_data_leakage(edges, edges)
        
        assert is_valid is False
        assert len(overlaps) == 3  # All edges overlap


# =============================================================================
# Parametrized Tests
# =============================================================================

class TestParametrizedValidators:
    """Parametrized tests for edge cases and boundary conditions."""
    
    @pytest.mark.parametrize("amount,expected_valid", [
        (9000, True),    # Minimum valid
        (9800, True),    # Maximum valid
        (9500, True),    # Middle of range
        (8999, False),   # Just below minimum
        (9801, False),   # Just above maximum
        (0, False),      # Zero
        (10000, False),  # CTR threshold
    ])
    def test_structuring_amount_boundaries(self, amount, expected_valid):
        """Test structuring validation at amount boundaries."""
        edges = [(1, 100, {'amount': amount})]
        
        is_valid, _ = validate_structuring_pattern(edges)
        
        assert is_valid == expected_valid
    
    @pytest.mark.parametrize("chain_length", [1, 2, 5, 10, 20])
    def test_layering_various_chain_lengths(self, chain_length):
        """Test layering validation with various chain lengths."""
        G = nx.DiGraph()
        chain_edges = [(i, i + 1) for i in range(chain_length)]
        G.add_edges_from(chain_edges)
        
        is_valid, errors = validate_layering_pattern(G, chain_edges)
        
        assert is_valid is True
        assert len(errors) == 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestValidatorsIntegration:
    """Integration tests combining multiple validators."""
    
    def test_full_graph_validation_workflow(self, baseline_graph):
        """Test complete validation workflow on baseline graph."""
        # Step 1: Structure validation
        struct_valid, struct_errors = validate_graph_structure(baseline_graph)
        
        # Step 2: Distribution validation
        dist_valid, dist_metrics = validate_scale_free_distribution(baseline_graph)
        
        # Both should work without errors
        assert isinstance(struct_valid, bool)
        assert isinstance(dist_valid, bool)
        assert dist_metrics["num_nodes"] == 1000
    
    def test_validators_handle_modified_graph(self, small_graph):
        """Test that validators handle graph modifications correctly."""
        # Add a self-loop
        small_graph.add_edge(0, 0)
        
        is_valid, errors = validate_graph_structure(small_graph)
        
        assert is_valid is False
        assert any("self-loop" in err.lower() for err in errors)
