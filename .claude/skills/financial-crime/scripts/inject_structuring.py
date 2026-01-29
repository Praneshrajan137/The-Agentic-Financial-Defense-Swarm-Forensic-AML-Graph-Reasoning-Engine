#!/usr/bin/env python3
"""
Inject Structuring (Smurfing) Crime Pattern
============================================
Injects fan-in pattern: multiple sources sending small amounts
to a single mule account, all below the CTR threshold.

Usage:
    python inject_structuring.py --graph baseline.pkl --mule node_42 --output poisoned.pkl
    python inject_structuring.py --graph baseline.pkl --output poisoned.pkl --auto-select-mule
"""

import argparse
import json
import pickle
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import networkx as nx
from faker import Faker


# Constants
CTR_THRESHOLD = 10000
STRUCTURING_MIN_AMOUNT = 9000.0
STRUCTURING_MAX_AMOUNT = 9800.0
STRUCTURING_NUM_SOURCES = 20
STRUCTURING_TIME_WINDOW_HOURS = 48


def inject_structuring(
    G: nx.DiGraph,
    mule_id: Optional[str] = None,
    num_sources: int = STRUCTURING_NUM_SOURCES,
    min_amount: float = STRUCTURING_MIN_AMOUNT,
    max_amount: float = STRUCTURING_MAX_AMOUNT,
    time_window_hours: int = STRUCTURING_TIME_WINDOW_HOURS,
    seed: Optional[int] = None
) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """
    Inject structuring (smurfing) crime pattern into graph.
    
    Creates a fan-in pattern where multiple source nodes send
    small amounts (below CTR threshold) to a single mule node.
    
    Args:
        G: Baseline financial network graph
        mule_id: Target mule node ID (auto-selected if None)
        num_sources: Number of smurf source nodes to create
        min_amount: Minimum transfer amount
        max_amount: Maximum transfer amount
        time_window_hours: Time window for all transfers
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (modified graph, ground truth metadata)
    
    Raises:
        ValueError: If mule_id specified but not in graph
        ValueError: If amounts exceed CTR threshold
    """
    # Validate amounts
    if min_amount >= CTR_THRESHOLD or max_amount >= CTR_THRESHOLD:
        raise ValueError(f"Amounts must be below CTR threshold (${CTR_THRESHOLD})")
    
    # Set seed
    if seed is not None:
        random.seed(seed)
    
    # Initialize Faker
    fake = Faker(['en_US', 'en_GB', 'en_IN'])
    if seed is not None:
        Faker.seed(seed)
    
    # Auto-select mule if not specified (prefer high in-degree nodes)
    if mule_id is None:
        # Select node with high in-degree (more realistic mule)
        in_degrees = dict(G.in_degree())
        sorted_nodes = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
        # Pick from top 10% of nodes by in-degree
        top_nodes = sorted_nodes[:max(1, len(sorted_nodes) // 10)]
        mule_node = random.choice(top_nodes)[0]
        mule_id = f"node_{mule_node}" if not str(mule_node).startswith("node_") else str(mule_node)
    else:
        # Validate mule exists
        mule_node = None
        for node in G.nodes():
            node_id = G.nodes[node].get('id', f"node_{node}")
            if node_id == mule_id or str(node) == mule_id:
                mule_node = node
                break
        
        if mule_node is None:
            raise ValueError(f"Mule node not found: {mule_id}")
    
    # Generate base timestamp
    base_time = datetime.now()
    
    # Ground truth structure
    ground_truth = {
        'crime_type': 'structuring',
        'mule_id': mule_id,
        'mule_node': mule_node,
        'sources': [],
        'total_amount': 0.0,
        'transaction_count': num_sources,
        'timestamp_range': {
            'start': None,
            'end': None
        }
    }
    
    timestamps = []
    
    # Create source nodes and edges
    for i in range(num_sources):
        # Create smurf node ID
        smurf_id = f"smurf_{i}"
        
        # Add smurf node with attributes
        G.add_node(smurf_id)
        G.nodes[smurf_id]['id'] = smurf_id
        G.nodes[smurf_id]['entity_type'] = 'person'
        G.nodes[smurf_id]['name'] = fake.name()
        G.nodes[smurf_id]['address'] = fake.address().replace('\n', ', ')
        G.nodes[smurf_id]['country'] = fake.country_code()
        G.nodes[smurf_id]['swift'] = fake.swift()
        G.nodes[smurf_id]['iban'] = fake.iban()
        G.nodes[smurf_id]['account_number'] = fake.bban()
        G.nodes[smurf_id]['risk_score'] = round(random.uniform(0.3, 0.7), 2)
        G.nodes[smurf_id]['verification_status'] = 'verified'
        G.nodes[smurf_id]['created_at'] = (
            base_time - timedelta(days=random.randint(30, 365))
        ).isoformat()
        G.nodes[smurf_id]['type'] = 'smurf'  # Crime indicator
        
        # Generate transfer amount (below CTR)
        amount = round(random.uniform(min_amount, max_amount), 2)
        
        # Generate timestamp within window
        timestamp = base_time + timedelta(
            seconds=random.randint(0, time_window_hours * 3600)
        )
        timestamps.append(timestamp)
        
        # Add edge from smurf to mule
        G.add_edge(smurf_id, mule_node)
        G.edges[smurf_id, mule_node]['transaction_id'] = f"txn_struct_{i}_{fake.uuid4()[:8]}"
        G.edges[smurf_id, mule_node]['amount'] = amount
        G.edges[smurf_id, mule_node]['currency'] = 'USD'
        G.edges[smurf_id, mule_node]['timestamp'] = timestamp.isoformat()
        G.edges[smurf_id, mule_node]['transaction_type'] = random.choice(['wire', 'ach', 'cash'])
        G.edges[smurf_id, mule_node]['label'] = 'structuring'
        G.edges[smurf_id, mule_node]['memo'] = fake.sentence(nb_words=3)
        
        # Record in ground truth
        ground_truth['sources'].append({
            'id': smurf_id,
            'amount': amount,
            'timestamp': timestamp.isoformat()
        })
        ground_truth['total_amount'] += amount
    
    # Set timestamp range
    ground_truth['timestamp_range']['start'] = min(timestamps).isoformat()
    ground_truth['timestamp_range']['end'] = max(timestamps).isoformat()
    ground_truth['total_amount'] = round(ground_truth['total_amount'], 2)
    
    return G, ground_truth


def save_results(
    G: nx.DiGraph,
    ground_truth: Dict[str, Any],
    output_path: str
) -> None:
    """
    Save graph and ground truth to files.
    
    Args:
        G: Modified graph with crime injected
        ground_truth: Crime metadata
        output_path: Path for pickle file (JSON saved alongside)
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save graph
    with open(path, 'wb') as f:
        pickle.dump(G, f)
    
    # Save ground truth
    json_path = path.with_suffix('.ground_truth.json')
    with open(json_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"Graph saved to {path}")
    print(f"Ground truth saved to {json_path}")
    print(f"  Mule ID: {ground_truth['mule_id']}")
    print(f"  Sources: {ground_truth['transaction_count']}")
    print(f"  Total Amount: ${ground_truth['total_amount']:,.2f}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Inject structuring (smurfing) crime pattern into graph'
    )
    parser.add_argument(
        '--graph', '-g',
        type=str,
        required=True,
        help='Input graph pickle file'
    )
    parser.add_argument(
        '--mule', '-m',
        type=str,
        default=None,
        help='Target mule node ID (auto-selected if not specified)'
    )
    parser.add_argument(
        '--num-sources', '-n',
        type=int,
        default=STRUCTURING_NUM_SOURCES,
        help=f'Number of smurf sources (default: {STRUCTURING_NUM_SOURCES})'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output pickle file path'
    )
    
    args = parser.parse_args()
    
    # Load graph
    print(f"Loading graph from {args.graph}...")
    with open(args.graph, 'rb') as f:
        G = pickle.load(f)
    
    print(f"Injecting structuring pattern...")
    G, ground_truth = inject_structuring(
        G,
        mule_id=args.mule,
        num_sources=args.num_sources,
        seed=args.seed
    )
    
    save_results(G, ground_truth, args.output)


if __name__ == '__main__':
    main()
