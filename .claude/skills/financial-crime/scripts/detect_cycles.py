#!/usr/bin/env python3
"""
Detect Cycles in Financial Graph
================================
Validates that graphs (especially after crime injection)
don't contain unintended cycles.

Usage:
    python detect_cycles.py --graph poisoned.pkl
    python detect_cycles.py --graph poisoned.pkl --crime-only
"""

import argparse
import pickle
from pathlib import Path
from typing import Any, List, Optional, Tuple

import networkx as nx


def detect_cycles(
    G: nx.DiGraph,
    crime_labels: Optional[List[str]] = None
) -> List[List[Any]]:
    """
    Detect all cycles in graph.
    
    Args:
        G: NetworkX DiGraph to analyze
        crime_labels: If specified, only check edges with these labels
    
    Returns:
        List of cycles (each cycle is a list of nodes)
    """
    if crime_labels:
        # Filter to only crime edges
        crime_edges = [
            (u, v) for u, v, data in G.edges(data=True)
            if data.get('label') in crime_labels
        ]
        if not crime_edges:
            return []
        subgraph = G.edge_subgraph(crime_edges)
    else:
        subgraph = G
    
    cycles = []
    try:
        # Find all simple cycles
        cycles = list(nx.simple_cycles(subgraph))
    except nx.NetworkXError:
        pass
    
    return cycles


def detect_crime_cycles(G: nx.DiGraph) -> List[List[Any]]:
    """
    Detect cycles specifically in crime-related edges.
    
    Args:
        G: NetworkX DiGraph with crime labels
    
    Returns:
        List of cycles in crime subgraph
    """
    return detect_cycles(G, crime_labels=['structuring', 'layering'])


def validate_layering_chain(
    G: nx.DiGraph,
    chain_nodes: List[Any]
) -> Tuple[bool, str]:
    """
    Validate that a layering chain has no cycles.
    
    Args:
        G: Graph containing the chain
        chain_nodes: Ordered list of nodes in the chain
    
    Returns:
        Tuple of (is_valid, message)
    """
    # Check for duplicate nodes (would indicate a cycle)
    if len(chain_nodes) != len(set(chain_nodes)):
        duplicates = [n for n in chain_nodes if chain_nodes.count(n) > 1]
        return False, f"Chain contains duplicate nodes: {duplicates}"
    
    # Check that chain forms a simple path
    for i in range(len(chain_nodes) - 1):
        if not G.has_edge(chain_nodes[i], chain_nodes[i + 1]):
            return False, f"Missing edge: {chain_nodes[i]} -> {chain_nodes[i + 1]}"
    
    # Check for back edges
    for i, node in enumerate(chain_nodes):
        for j in range(i + 2, len(chain_nodes)):
            if G.has_edge(chain_nodes[j], node):
                return False, f"Back edge detected: {chain_nodes[j]} -> {node}"
    
    return True, "Chain is valid (no cycles)"


def get_graph_stats(G: nx.DiGraph) -> dict:
    """
    Get statistics about graph structure.
    
    Args:
        G: NetworkX DiGraph
    
    Returns:
        Dictionary with graph statistics
    """
    stats = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'is_dag': nx.is_directed_acyclic_graph(G),
        'strongly_connected_components': nx.number_strongly_connected_components(G),
        'weakly_connected_components': nx.number_weakly_connected_components(G)
    }
    
    # Count edges by label
    label_counts = {}
    for _, _, data in G.edges(data=True):
        label = data.get('label', 'unknown')
        label_counts[label] = label_counts.get(label, 0) + 1
    stats['edge_labels'] = label_counts
    
    return stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Detect cycles in financial graph'
    )
    parser.add_argument(
        '--graph', '-g',
        type=str,
        required=True,
        help='Input graph pickle file'
    )
    parser.add_argument(
        '--crime-only',
        action='store_true',
        help='Only check crime-labeled edges'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )
    
    args = parser.parse_args()
    
    # Load graph
    print(f"Loading graph from {args.graph}...")
    with open(args.graph, 'rb') as f:
        G = pickle.load(f)
    
    # Get stats
    stats = get_graph_stats(G)
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {stats['nodes']}")
    print(f"  Edges: {stats['edges']}")
    print(f"  Is DAG: {stats['is_dag']}")
    print(f"  Strongly Connected Components: {stats['strongly_connected_components']}")
    print(f"  Edge Labels: {stats['edge_labels']}")
    
    # Detect cycles
    if args.crime_only:
        print(f"\nChecking for cycles in crime edges only...")
        cycles = detect_crime_cycles(G)
    else:
        print(f"\nChecking for cycles in entire graph...")
        cycles = detect_cycles(G)
    
    if cycles:
        print(f"\nWARNING: Found {len(cycles)} cycle(s)!")
        if args.verbose:
            for i, cycle in enumerate(cycles[:10]):  # Show first 10
                print(f"  Cycle {i+1}: {' -> '.join(str(n) for n in cycle)}")
            if len(cycles) > 10:
                print(f"  ... and {len(cycles) - 10} more")
        return 1
    else:
        print(f"\nSUCCESS: No cycles detected")
        return 0


if __name__ == '__main__':
    exit(main())
