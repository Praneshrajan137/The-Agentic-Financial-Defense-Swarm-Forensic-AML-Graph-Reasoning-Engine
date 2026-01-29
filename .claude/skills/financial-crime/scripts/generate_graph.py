#!/usr/bin/env python3
"""
Generate Scale-Free Financial Networks
======================================
Uses NetworkX to create realistic financial transaction graphs
with Faker-generated entity attributes.

Usage:
    python generate_graph.py --nodes 1000 --output baseline.pkl
    python generate_graph.py --nodes 1000 --seed 42 --output baseline.pkl
"""

import argparse
import pickle
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

import networkx as nx
from faker import Faker


# Constants
DEFAULT_ALPHA = 0.41
DEFAULT_BETA = 0.54
DEFAULT_GAMMA = 0.05
DEFAULT_NODES = 1000
LOCALES = ['en_US', 'en_GB', 'en_IN']


def generate_scale_free_graph(
    n: int,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    seed: Optional[int] = None,
    locales: Optional[List[str]] = None
) -> nx.DiGraph:
    """
    Generate scale-free graph with realistic entity attributes.
    
    Args:
        n: Number of nodes
        alpha: Probability of adding edge based on in-degree
        beta: Probability of adding edge between existing nodes
        gamma: Probability of adding edge based on out-degree
        seed: Random seed for reproducibility
        locales: Faker locales for entity generation
    
    Returns:
        Directed graph with node and edge attributes
    
    Raises:
        ValueError: If alpha + beta + gamma != 1.0
    """
    # Validate parameters
    if abs(alpha + beta + gamma - 1.0) > 1e-9:
        raise ValueError(f"alpha + beta + gamma must equal 1.0, got {alpha + beta + gamma}")
    
    # Set seeds for reproducibility
    if seed is not None:
        random.seed(seed)
    
    # Initialize Faker with locales
    if locales is None:
        locales = LOCALES
    fake = Faker(locales)
    if seed is not None:
        Faker.seed(seed)
    
    # Generate scale-free graph structure
    G = nx.scale_free_graph(n, alpha=alpha, beta=beta, gamma=gamma, seed=seed)
    
    # Convert to simple DiGraph (remove multi-edges)
    G = nx.DiGraph(G)
    
    # Add node attributes
    entity_types = ['person', 'company', 'bank']
    entity_weights = [0.7, 0.25, 0.05]  # 70% people, 25% companies, 5% banks
    
    for node in G.nodes():
        entity_type = random.choices(entity_types, weights=entity_weights)[0]
        
        G.nodes[node]['id'] = f"node_{node}"
        G.nodes[node]['entity_type'] = entity_type
        
        if entity_type == 'person':
            G.nodes[node]['name'] = fake.name()
            G.nodes[node]['company'] = None
        elif entity_type == 'company':
            G.nodes[node]['name'] = fake.company()
            G.nodes[node]['company'] = G.nodes[node]['name']
        else:  # bank
            G.nodes[node]['name'] = fake.company() + " Bank"
            G.nodes[node]['company'] = G.nodes[node]['name']
        
        G.nodes[node]['address'] = fake.address().replace('\n', ', ')
        G.nodes[node]['country'] = fake.country_code()
        G.nodes[node]['swift'] = fake.swift()
        G.nodes[node]['iban'] = fake.iban()
        G.nodes[node]['account_number'] = fake.bban()
        G.nodes[node]['risk_score'] = round(random.uniform(0.0, 1.0), 2)
        G.nodes[node]['verification_status'] = random.choice(['verified', 'pending', 'verified', 'verified'])
        G.nodes[node]['created_at'] = fake.date_time_between(
            start_date='-2y',
            end_date='now'
        ).isoformat()
    
    # Add edge attributes
    base_time = datetime.now() - timedelta(days=365)
    
    for u, v in G.edges():
        G.edges[u, v]['transaction_id'] = f"txn_{fake.uuid4()[:8]}"
        G.edges[u, v]['amount'] = round(random.uniform(100, 50000), 2)
        G.edges[u, v]['currency'] = 'USD'
        G.edges[u, v]['timestamp'] = (
            base_time + timedelta(days=random.randint(0, 365))
        ).isoformat()
        G.edges[u, v]['transaction_type'] = random.choice(['wire', 'ach', 'cash', 'internal'])
        G.edges[u, v]['label'] = 'legitimate'
        G.edges[u, v]['memo'] = fake.sentence(nb_words=4)
    
    return G


def save_graph(G: nx.DiGraph, output_path: str) -> None:
    """
    Save graph to pickle file.
    
    Args:
        G: NetworkX DiGraph
        output_path: Path to save pickle file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(G, f)
    
    print(f"Graph saved to {path}")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate scale-free financial network graph'
    )
    parser.add_argument(
        '--nodes', '-n',
        type=int,
        default=DEFAULT_NODES,
        help=f'Number of nodes (default: {DEFAULT_NODES})'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=DEFAULT_ALPHA,
        help=f'Alpha parameter (default: {DEFAULT_ALPHA})'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=DEFAULT_BETA,
        help=f'Beta parameter (default: {DEFAULT_BETA})'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=DEFAULT_GAMMA,
        help=f'Gamma parameter (default: {DEFAULT_GAMMA})'
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
    
    print(f"Generating scale-free graph with {args.nodes} nodes...")
    G = generate_scale_free_graph(
        n=args.nodes,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        seed=args.seed
    )
    
    save_graph(G, args.output)


if __name__ == '__main__':
    main()
