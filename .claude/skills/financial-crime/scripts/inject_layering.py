#!/usr/bin/env python3
"""
Inject Layering Crime Pattern
==============================
Injects chain pattern: directed path with value decay
to obscure money trail through multiple intermediaries.

Usage:
    python inject_layering.py --graph baseline.pkl --source node_10 --dest node_99 --output poisoned.pkl
    python inject_layering.py --graph baseline.pkl --output poisoned.pkl --auto-select
"""

import argparse
import json
import pickle
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from faker import Faker


# Constants
LAYERING_CHAIN_LENGTH_MIN = 5
LAYERING_CHAIN_LENGTH_MAX = 7
LAYERING_MIN_DECAY = 0.02
LAYERING_MAX_DECAY = 0.05
LAYERING_INITIAL_AMOUNT = 100000.0
LAYERING_TIME_WINDOW_HOURS = 24


def inject_layering(
    G: nx.DiGraph,
    source: Optional[str] = None,
    dest: Optional[str] = None,
    chain_length: Optional[int] = None,
    initial_amount: float = LAYERING_INITIAL_AMOUNT,
    min_decay: float = LAYERING_MIN_DECAY,
    max_decay: float = LAYERING_MAX_DECAY,
    time_window_hours: int = LAYERING_TIME_WINDOW_HOURS,
    seed: Optional[int] = None
) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """
    Inject layering crime pattern into graph.
    
    Creates a chain of transfers with decay to obscure the money trail.
    
    Args:
        G: Baseline financial network graph
        source: Source node ID (auto-selected if None)
        dest: Destination node ID (auto-selected if None)
        chain_length: Number of intermediary hops (random 5-7 if None)
        initial_amount: Starting amount
        min_decay: Minimum decay rate per hop
        max_decay: Maximum decay rate per hop
        time_window_hours: Time window for all transfers
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (modified graph, ground truth metadata)
    
    Raises:
        ValueError: If source or dest not in graph (when specified)
        ValueError: If chain would create cycles
    """
    # Set seed
    if seed is not None:
        random.seed(seed)
    
    # Initialize Faker
    fake = Faker(['en_US', 'en_GB', 'en_IN'])
    if seed is not None:
        Faker.seed(seed)
    
    # Determine chain length
    if chain_length is None:
        chain_length = random.randint(LAYERING_CHAIN_LENGTH_MIN, LAYERING_CHAIN_LENGTH_MAX)
    
    # Get existing node list
    existing_nodes = list(G.nodes())
    
    # Auto-select source if not specified
    if source is None:
        source_node = random.choice(existing_nodes)
        source = G.nodes[source_node].get('id', f"node_{source_node}")
    else:
        source_node = _find_node(G, source)
        if source_node is None:
            raise ValueError(f"Source node not found: {source}")
    
    # Auto-select dest if not specified (different from source)
    if dest is None:
        available_nodes = [n for n in existing_nodes if n != source_node]
        dest_node = random.choice(available_nodes)
        dest = G.nodes[dest_node].get('id', f"node_{dest_node}")
    else:
        dest_node = _find_node(G, dest)
        if dest_node is None:
            raise ValueError(f"Destination node not found: {dest}")
    
    # Generate base timestamp
    base_time = datetime.now()
    
    # Build chain: source -> layer_0 -> layer_1 -> ... -> dest
    chain = [source_node]
    
    # Create intermediate layer nodes
    for i in range(chain_length):
        layer_id = f"layer_{i}"
        
        # Add layer node
        G.add_node(layer_id)
        G.nodes[layer_id]['id'] = layer_id
        G.nodes[layer_id]['entity_type'] = random.choice(['person', 'company'])
        G.nodes[layer_id]['name'] = fake.name() if G.nodes[layer_id]['entity_type'] == 'person' else fake.company()
        G.nodes[layer_id]['address'] = fake.address().replace('\n', ', ')
        G.nodes[layer_id]['country'] = fake.country_code()
        G.nodes[layer_id]['swift'] = fake.swift()
        G.nodes[layer_id]['iban'] = fake.iban()
        G.nodes[layer_id]['account_number'] = fake.bban()
        G.nodes[layer_id]['risk_score'] = round(random.uniform(0.2, 0.6), 2)
        G.nodes[layer_id]['verification_status'] = 'verified'
        G.nodes[layer_id]['created_at'] = (
            base_time - timedelta(days=random.randint(60, 730))
        ).isoformat()
        G.nodes[layer_id]['type'] = 'layer'  # Crime indicator
        
        chain.append(layer_id)
    
    # Add destination to chain
    chain.append(dest_node)
    
    # Ground truth structure
    ground_truth = {
        'crime_type': 'layering',
        'source_id': source,
        'dest_id': dest,
        'chain': [],
        'chain_length': chain_length,
        'initial_amount': initial_amount,
        'final_amount': 0.0,
        'total_decay': 0.0,
        'timestamp_range': {
            'start': None,
            'end': None
        }
    }
    
    # Create edges along chain
    current_amount = initial_amount
    timestamps = []
    
    for i in range(len(chain) - 1):
        from_node = chain[i]
        to_node = chain[i + 1]
        
        # Calculate decay for this hop
        decay = random.uniform(min_decay, max_decay)
        amount_after_decay = current_amount * (1 - decay)
        
        # Generate timestamp (progressive within window)
        hop_delay_minutes = random.randint(10, 120)
        timestamp = base_time + timedelta(minutes=i * hop_delay_minutes + random.randint(0, 30))
        timestamps.append(timestamp)
        
        # Add edge
        G.add_edge(from_node, to_node)
        G.edges[from_node, to_node]['transaction_id'] = f"txn_layer_{i}_{fake.uuid4()[:8]}"
        G.edges[from_node, to_node]['amount'] = round(current_amount, 2)
        G.edges[from_node, to_node]['currency'] = 'USD'
        G.edges[from_node, to_node]['timestamp'] = timestamp.isoformat()
        G.edges[from_node, to_node]['transaction_type'] = random.choice(['wire', 'internal'])
        G.edges[from_node, to_node]['label'] = 'layering'
        G.edges[from_node, to_node]['memo'] = fake.sentence(nb_words=3)
        G.edges[from_node, to_node]['decay_rate'] = round(decay, 4)
        
        # Record in ground truth
        ground_truth['chain'].append({
            'from': G.nodes[from_node].get('id', str(from_node)),
            'to': G.nodes[to_node].get('id', str(to_node)),
            'amount': round(current_amount, 2),
            'decay_rate': round(decay, 4),
            'timestamp': timestamp.isoformat()
        })
        
        # Apply decay for next hop
        current_amount = amount_after_decay
    
    # Set final values
    ground_truth['final_amount'] = round(current_amount, 2)
    ground_truth['total_decay'] = round(1 - (current_amount / initial_amount), 4)
    ground_truth['timestamp_range']['start'] = min(timestamps).isoformat()
    ground_truth['timestamp_range']['end'] = max(timestamps).isoformat()
    
    # Validate no cycles in crime subgraph
    crime_edges = [(chain[i], chain[i+1]) for i in range(len(chain)-1)]
    if not validate_no_cycles(G, crime_edges):
        raise ValueError("Layering chain would create cycles")
    
    return G, ground_truth


def _find_node(G: nx.DiGraph, node_id: str) -> Optional[Any]:
    """Find node by ID in graph."""
    for node in G.nodes():
        if G.nodes[node].get('id') == node_id or str(node) == node_id:
            return node
    return None


def validate_no_cycles(G: nx.DiGraph, crime_edges: List[Tuple[Any, Any]]) -> bool:
    """
    Validate that crime edges don't create cycles.
    
    Args:
        G: Graph to check
        crime_edges: List of edges to validate
    
    Returns:
        True if no cycles, False otherwise
    """
    # Create subgraph with crime edges
    subgraph = G.edge_subgraph(crime_edges)
    
    try:
        nx.find_cycle(subgraph)
        return False  # Cycle found
    except nx.NetworkXNoCycle:
        return True  # No cycles


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
        output_path: Path for pickle file
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
    print(f"  Source: {ground_truth['source_id']}")
    print(f"  Destination: {ground_truth['dest_id']}")
    print(f"  Chain Length: {ground_truth['chain_length']}")
    print(f"  Initial Amount: ${ground_truth['initial_amount']:,.2f}")
    print(f"  Final Amount: ${ground_truth['final_amount']:,.2f}")
    print(f"  Total Decay: {ground_truth['total_decay']*100:.2f}%")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Inject layering crime pattern into graph'
    )
    parser.add_argument(
        '--graph', '-g',
        type=str,
        required=True,
        help='Input graph pickle file'
    )
    parser.add_argument(
        '--source', '-s',
        type=str,
        default=None,
        help='Source node ID (auto-selected if not specified)'
    )
    parser.add_argument(
        '--dest', '-d',
        type=str,
        default=None,
        help='Destination node ID (auto-selected if not specified)'
    )
    parser.add_argument(
        '--chain-length', '-l',
        type=int,
        default=None,
        help=f'Chain length (default: random {LAYERING_CHAIN_LENGTH_MIN}-{LAYERING_CHAIN_LENGTH_MAX})'
    )
    parser.add_argument(
        '--initial-amount', '-a',
        type=float,
        default=LAYERING_INITIAL_AMOUNT,
        help=f'Initial amount (default: ${LAYERING_INITIAL_AMOUNT:,.2f})'
    )
    parser.add_argument(
        '--seed',
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
    
    print(f"Injecting layering pattern...")
    G, ground_truth = inject_layering(
        G,
        source=args.source,
        dest=args.dest,
        chain_length=args.chain_length,
        initial_amount=args.initial_amount,
        seed=args.seed
    )
    
    save_results(G, ground_truth, args.output)


if __name__ == '__main__':
    main()
