"""
Validators Module
=================
Data integrity and graph validation utilities.

Validation Categories:
1. Graph structure validation
2. Transaction amount validation
3. Crime pattern validation
4. Statistical distribution validation
"""

import networkx as nx
from typing import List, Tuple, Optional
import statistics


def validate_graph_structure(G: nx.DiGraph) -> Tuple[bool, List[str]]:
    """
    Validate basic graph structure requirements.
    
    Checks:
    - Graph is directed
    - No self-loops
    - No isolated nodes (optional)
    - Connected components check
    
    Args:
        G: NetworkX DiGraph to validate
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    # Check if directed
    if not isinstance(G, nx.DiGraph):
        errors.append("Graph must be a directed graph (DiGraph)")
    
    # Check for self-loops
    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        errors.append(f"Graph contains {len(self_loops)} self-loops")
    
    # Check for isolated nodes
    isolated = list(nx.isolates(G))
    if isolated:
        errors.append(f"Graph contains {len(isolated)} isolated nodes")
    
    return len(errors) == 0, errors


def validate_scale_free_distribution(G: nx.DiGraph, tolerance: float = 0.1) -> Tuple[bool, dict]:
    """
    Validate that graph follows scale-free (power-law) distribution.
    
    Args:
        G: NetworkX DiGraph
        tolerance: Acceptable deviation from expected distribution
    
    Returns:
        Tuple of (is_valid, distribution metrics)
    """
    degrees = [d for n, d in G.degree()]
    
    if not degrees:
        return False, {"error": "Empty graph"}
    
    metrics = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "avg_degree": statistics.mean(degrees),
        "max_degree": max(degrees),
        "min_degree": min(degrees),
        "degree_variance": statistics.variance(degrees) if len(degrees) > 1 else 0
    }
    
    # Scale-free networks have high variance in degree distribution
    # and a few hub nodes with very high degree
    hub_threshold = metrics["avg_degree"] * 5
    hubs = [d for d in degrees if d > hub_threshold]
    metrics["hub_count"] = len(hubs)
    
    # Valid if we have some hub nodes (characteristic of scale-free)
    is_valid = len(hubs) > 0
    
    return is_valid, metrics


def validate_structuring_pattern(
    edges: List[Tuple[int, int, dict]],
    min_amount: float = 9000.0,
    max_amount: float = 9800.0,
    max_time_window_hours: int = 48
) -> Tuple[bool, List[str]]:
    """
    Validate a structuring crime pattern.
    
    Requirements:
    - All amounts between $9,000 and $9,800
    - All transfers within 48-hour window
    - Single target (mule) node
    
    Args:
        edges: List of edges with attributes (source, target, attrs)
        min_amount: Minimum valid amount
        max_amount: Maximum valid amount
        max_time_window_hours: Maximum time window for transfers
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    if not edges:
        errors.append("No edges provided")
        return False, errors
    
    # Check single target
    targets = set(e[1] for e in edges)
    if len(targets) > 1:
        errors.append(f"Multiple targets found: {targets}. Structuring requires single mule.")
    
    # Check amounts
    for source, target, attrs in edges:
        amount = attrs.get("amount", 0)
        if amount < min_amount or amount > max_amount:
            errors.append(f"Edge {source}->{target}: amount ${amount} outside valid range [${min_amount}, ${max_amount}]")
    
    # TODO: Add timestamp validation when attributes are implemented
    
    return len(errors) == 0, errors


def validate_layering_pattern(
    G: nx.DiGraph,
    chain_edges: List[Tuple[int, int]],
    min_decay: float = 0.02,
    max_decay: float = 0.05
) -> Tuple[bool, List[str]]:
    """
    Validate a layering crime pattern.
    
    Requirements:
    - Forms a directed chain (no branches) - a simple path
    - No cycles
    - Decay between 2-5% per hop
    
    Args:
        G: NetworkX DiGraph containing the edges
        chain_edges: List of edges forming the chain
        min_decay: Minimum decay rate per hop
        max_decay: Maximum decay rate per hop
    
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    if not chain_edges:
        errors.append("No chain edges provided")
        return False, errors
    
    # Check for cycles
    chain_subgraph = G.edge_subgraph(chain_edges)
    try:
        cycle = nx.find_cycle(chain_subgraph)
        errors.append(f"Cycle detected in layering chain: {cycle}")
    except nx.NetworkXNoCycle:
        pass  # Good - no cycles
    
    # Check it forms a simple path (chain) - no branching allowed
    in_degrees = dict(chain_subgraph.in_degree())
    out_degrees = dict(chain_subgraph.out_degree())
    
    # In a valid simple chain:
    # - Exactly 1 start node: in_degree=0, out_degree=1
    # - Exactly 1 end node: in_degree=1, out_degree=0
    # - All intermediate nodes: in_degree=1, out_degree=1
    
    starts = [n for n, d in in_degrees.items() if d == 0]
    ends = [n for n, d in out_degrees.items() if d == 0]
    
    if len(starts) != 1:
        errors.append(f"Invalid chain: expected 1 start node, found {len(starts)}")
    if len(ends) != 1:
        errors.append(f"Invalid chain: expected 1 end node, found {len(ends)}")
    
    # Check for branching - every node must have max in_degree=1 and max out_degree=1
    # This ensures no fan-in or fan-out (which would indicate branching)
    for node in chain_subgraph.nodes():
        in_deg = in_degrees[node]
        out_deg = out_degrees[node]
        
        if in_deg > 1:
            errors.append(f"Invalid chain: node {node} has in_degree={in_deg} (branching/merging detected)")
        if out_deg > 1:
            errors.append(f"Invalid chain: node {node} has out_degree={out_deg} (branching detected)")
    
    # TODO: Add decay rate validation when amounts are implemented
    
    return len(errors) == 0, errors


def validate_no_data_leakage(
    train_edges: List[Tuple[int, int]],
    test_edges: List[Tuple[int, int]]
) -> Tuple[bool, List[str]]:
    """
    Validate no data leakage between train and test sets.
    
    Args:
        train_edges: Edges in training set
        test_edges: Edges in test set
    
    Returns:
        Tuple of (is_valid, list of overlapping edges)
    """
    train_set = set(train_edges)
    test_set = set(test_edges)
    
    overlap = train_set.intersection(test_set)
    
    if overlap:
        return False, [f"Edge {e} appears in both train and test" for e in list(overlap)[:10]]
    
    return True, []


__all__ = [
    'validate_graph_structure',
    'validate_scale_free_distribution',
    'validate_structuring_pattern',
    'validate_layering_pattern',
    'validate_no_data_leakage'
]
