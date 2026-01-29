"""
Graph Generator Module
======================
Generates scale-free financial transaction graphs using NetworkX.

Technical Specification:
- Algorithm: Barabási-Albert scale-free graph
- Parameters: alpha=0.41, beta=0.54, gamma=0.05
- Target: 1,000 nodes, 10,000 edges
- Performance: < 10 seconds generation time
"""

import networkx as nx
from typing import Optional, List, Union
import random
import uuid
import pickle
import logging
from datetime import datetime, timedelta
from faker import Faker

logger = logging.getLogger(__name__)


def generate_scale_free_graph(
    n_nodes: int = 1000,
    alpha: float = 0.41,
    beta: float = 0.54,
    gamma: float = 0.05,
    seed: Optional[int] = None
) -> nx.DiGraph:
    """
    Generate a scale-free directed graph representing a financial network.
    
    Args:
        n_nodes: Number of nodes (entities) in the graph
        alpha: Probability for adding a new node connected to an existing node
        beta: Probability for adding an edge between two existing nodes
        gamma: Probability for adding a new node connected from an existing node
        seed: Random seed for reproducibility
    
    Returns:
        NetworkX DiGraph representing the financial network
    
    Raises:
        ValueError: If alpha + beta + gamma != 1.0
    """
    # Validate parameters sum to 1.0
    if abs(alpha + beta + gamma - 1.0) > 1e-9:
        raise ValueError(f"alpha + beta + gamma must equal 1.0, got {alpha + beta + gamma}")
    
    if seed is not None:
        random.seed(seed)
    
    # Generate scale-free graph
    G = nx.scale_free_graph(n=n_nodes, alpha=alpha, beta=beta, gamma=gamma, seed=seed)
    
    return G


def add_entity_attributes(
    G: nx.DiGraph, 
    faker_instance=None, 
    locales: Optional[List[str]] = None,
    seed: Optional[int] = None
) -> nx.DiGraph:
    """
    Add entity attributes to graph nodes using Faker.
    
    Args:
        G: NetworkX DiGraph
        faker_instance: Optional Faker instance for entity generation
        locales: List of Faker locales for entity generation (default: ['en_US', 'en_GB', 'en_IN'])
        seed: Random seed for reproducibility
    
    Returns:
        Graph with entity attributes added to nodes
    """
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)
    
    if faker_instance is None:
        locales = locales or ['en_US', 'en_GB', 'en_IN']
        faker_instance = Faker(locales)
    
    entity_types = ['person', 'company', 'bank']
    entity_weights = [0.7, 0.25, 0.05]
    
    for node in G.nodes():
        entity_type = random.choices(entity_types, weights=entity_weights)[0]
        G.nodes[node]['entity_type'] = entity_type
        
        if entity_type == 'person':
            G.nodes[node]['name'] = faker_instance.name()
        elif entity_type == 'company':
            G.nodes[node]['name'] = faker_instance.company()
        else:  # bank
            G.nodes[node]['name'] = f"{faker_instance.company()} Bank"
        
        G.nodes[node]['address'] = faker_instance.address().replace('\n', ', ')
        G.nodes[node]['swift'] = faker_instance.swift()
        G.nodes[node]['country'] = faker_instance.country_code()
        G.nodes[node]['risk_score'] = round(random.uniform(0, 1), 2)
        G.nodes[node]['verification_status'] = 'verified'
    
    return G


def add_transaction_attributes(
    G: nx.DiGraph,
    seed: Optional[int] = None,
    base_time: Optional[datetime] = None
) -> nx.DiGraph:
    """
    Add transaction attributes to graph edges.
    
    Args:
        G: NetworkX DiGraph or MultiDiGraph
        seed: Random seed for reproducibility
        base_time: Base timestamp for transactions (default: now)
    
    Returns:
        Graph with transaction attributes added to edges
    """
    if seed is not None:
        random.seed(seed)
    
    if base_time is None:
        base_time = datetime.now()
    
    transaction_types = ['wire', 'ach', 'cash', 'internal']
    
    # Handle both DiGraph and MultiDiGraph
    if isinstance(G, nx.MultiDiGraph):
        for u, v, key in G.edges(keys=True):
            G.edges[u, v, key]['transaction_id'] = f"txn_{uuid.uuid4().hex[:8]}"
            G.edges[u, v, key]['amount'] = round(random.uniform(100, 50000), 2)
            G.edges[u, v, key]['currency'] = 'USD'
            G.edges[u, v, key]['timestamp'] = base_time - timedelta(days=random.randint(0, 365))
            G.edges[u, v, key]['transaction_type'] = random.choice(transaction_types)
            G.edges[u, v, key]['label'] = 'legitimate'
            G.edges[u, v, key]['memo'] = None
    else:
        for u, v in G.edges():
            G.edges[u, v]['transaction_id'] = f"txn_{uuid.uuid4().hex[:8]}"
            G.edges[u, v]['amount'] = round(random.uniform(100, 50000), 2)
            G.edges[u, v]['currency'] = 'USD'
            G.edges[u, v]['timestamp'] = base_time - timedelta(days=random.randint(0, 365))
            G.edges[u, v]['transaction_type'] = random.choice(transaction_types)
            G.edges[u, v]['label'] = 'legitimate'
            G.edges[u, v]['memo'] = None
    
    return G


def save_graph(graph: nx.DiGraph, filepath: Union[str, 'Path']) -> None:
    """
    Save graph to pickle file.
    
    Args:
        graph: NetworkX graph to save
        filepath: Output file path
    """
    logger.info(f"Saving graph to {filepath}")
    with open(filepath, 'wb') as f:
        pickle.dump(graph, f)
    logger.info(f"Graph saved successfully ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")


def load_graph(filepath: Union[str, 'Path']) -> nx.DiGraph:
    """
    Load graph from pickle file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded NetworkX graph
    """
    logger.info(f"Loading graph from {filepath}")
    with open(filepath, 'rb') as f:
        graph = pickle.load(f)
    logger.info(f"Graph loaded successfully ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")
    return graph


__all__ = [
    'generate_scale_free_graph',
    'add_entity_attributes',
    'add_transaction_attributes',
    'save_graph',
    'load_graph'
]
