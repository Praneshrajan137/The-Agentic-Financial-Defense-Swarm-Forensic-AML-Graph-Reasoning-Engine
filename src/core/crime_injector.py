"""
Crime Injector Module
=====================
Surgically injects money laundering typologies into financial graphs.

Supported Crime Types:
1. Structuring (Smurfing): Fan-in pattern with multiple small deposits
2. Layering: Chain transfers with decay to obscure money trail

Technical Specifications:
- Structuring: 20 transfers to 1 mule, $9k-$9.8k each, 48hr window
- Layering: Directed chain with 2-5% decay per hop, no cycles
"""

import networkx as nx
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import random
import uuid
import json
import logging
from faker import Faker

logger = logging.getLogger(__name__)


@dataclass
class StructuringConfig:
    """Configuration for structuring crime injection."""
    num_sources: int = 20
    mule_node: Optional[int] = None
    min_amount: float = 9000.0
    max_amount: float = 9800.0
    time_window_hours: int = 48


@dataclass
class LayeringConfig:
    """Configuration for layering crime injection."""
    chain_length: int = 5
    min_decay: float = 0.02
    max_decay: float = 0.05
    initial_amount: float = 100000.0


@dataclass
class InjectedCrime:
    """Record of an injected crime for labeling."""
    crime_type: str
    nodes_involved: List[int]
    edges_involved: List[Tuple[int, int]]
    metadata: dict


def inject_structuring(
    G: nx.DiGraph,
    config: Optional[StructuringConfig] = None,
    seed: Optional[int] = None
) -> Tuple[nx.DiGraph, InjectedCrime]:
    """
    Inject structuring (smurfing) crime pattern into graph.
    
    Pattern: Fan-in - multiple sources sending small amounts to single mule.
    - 20 source nodes each send $9,000-$9,800 to 1 mule
    - All transfers within 48-hour window
    - Amounts stay below $10,000 CTR threshold
    
    Args:
        G: NetworkX DiGraph to inject crime into
        config: Structuring configuration parameters
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (modified graph, crime record)
    """
    if config is None:
        config = StructuringConfig()
    
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)
    
    fake = Faker()
    
    # Get the maximum node ID to create new unique nodes
    max_node = max(G.nodes()) if G.nodes() else 0
    
    # Select or use existing mule node
    if config.mule_node is not None:
        mule_id = config.mule_node
    else:
        # Select a random existing node as the mule
        mule_id = random.choice(list(G.nodes()))
    
    # Create source nodes and edges
    source_nodes = []
    edges_involved = []
    amounts = []
    base_time = datetime.now()
    
    for i in range(config.num_sources):
        source_id = max_node + 1 + i
        
        # Add source node with attributes
        G.add_node(
            source_id,
            name=fake.name(),
            entity_type='person',
            address=fake.address().replace('\n', ', '),
            swift=fake.swift(),
            country=fake.country_code(),
            risk_score=round(random.uniform(0.3, 0.7), 2),
            verification_status='verified'
        )
        source_nodes.append(source_id)
        
        # Generate amount below CTR threshold ($10,000)
        amount = round(random.uniform(config.min_amount, config.max_amount), 2)
        amounts.append(amount)
        
        # Generate timestamp within time window
        hours_offset = random.uniform(0, config.time_window_hours)
        timestamp = base_time + timedelta(hours=hours_offset)
        
        # Add edge from source to mule
        G.add_edge(
            source_id,
            mule_id,
            transaction_id=f"txn_{uuid.uuid4().hex[:8]}",
            amount=amount,
            currency='USD',
            timestamp=timestamp,
            transaction_type='cash',
            label='structuring',
            memo=fake.sentence(nb_words=4)
        )
        edges_involved.append((source_id, mule_id))
    
    # Create crime record
    crime = InjectedCrime(
        crime_type='structuring',
        nodes_involved=[mule_id] + source_nodes,
        edges_involved=edges_involved,
        metadata={
            'mule_id': mule_id,
            'source_count': config.num_sources,
            'total_amount': round(sum(amounts), 2),
            'time_window_hours': config.time_window_hours,
            'amounts': amounts
        }
    )
    
    return G, crime


def inject_layering(
    G: nx.DiGraph,
    config: Optional[LayeringConfig] = None,
    seed: Optional[int] = None,
    source_node: Optional[int] = None,
    dest_node: Optional[int] = None
) -> Tuple[nx.DiGraph, InjectedCrime]:
    """
    Inject layering crime pattern into graph.
    
    Pattern: Chain transfers with decay to obscure money trail.
    - Directed chain of transfers
    - 2-5% decay per hop
    - No cycles allowed
    
    Args:
        G: NetworkX DiGraph to inject crime into
        config: Layering configuration parameters
        seed: Random seed for reproducibility
        source_node: Optional starting node (uses random existing node if None)
        dest_node: Optional destination node (uses random existing node if None)
    
    Returns:
        Tuple of (modified graph, crime record)
    """
    if config is None:
        config = LayeringConfig()
    
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)
    
    fake = Faker()
    
    # Get the maximum node ID to create new unique nodes
    max_node = max(G.nodes()) if G.nodes() else 0
    existing_nodes = list(G.nodes())
    
    # Select source and destination nodes
    if source_node is None:
        source_node = random.choice(existing_nodes)
    if dest_node is None:
        # Ensure dest is different from source
        available_nodes = [n for n in existing_nodes if n != source_node]
        dest_node = random.choice(available_nodes) if available_nodes else source_node
    
    # Build chain: source -> intermediate_1 -> ... -> intermediate_n -> dest
    chain_nodes = [source_node]
    
    # Create intermediate nodes
    for i in range(config.chain_length):
        new_node = max_node + 1 + i
        
        # Add intermediate node (shell companies)
        G.add_node(
            new_node,
            name=fake.company(),
            entity_type='company',
            address=fake.address().replace('\n', ', '),
            swift=fake.swift(),
            country=fake.country_code(),
            risk_score=round(random.uniform(0.4, 0.8), 2),
            verification_status='verified'
        )
        chain_nodes.append(new_node)
    
    # Add destination to chain
    chain_nodes.append(dest_node)
    
    # Build edges with decay
    edges_involved = []
    amounts = []
    decays = []
    amount = config.initial_amount
    base_time = datetime.now()
    hop_interval_minutes = 30  # Rapid hops within 24-hour window
    
    for i in range(len(chain_nodes) - 1):
        src = chain_nodes[i]
        tgt = chain_nodes[i + 1]
        
        # Apply decay
        decay = random.uniform(config.min_decay, config.max_decay)
        decays.append(decay)
        
        if i > 0:  # Don't decay the first transfer
            amount = amount * (1 - decay)
        
        amounts.append(round(amount, 2))
        
        # Generate timestamp with rapid velocity
        timestamp = base_time + timedelta(minutes=hop_interval_minutes * i)
        
        # Add edge
        G.add_edge(
            src,
            tgt,
            transaction_id=f"txn_{uuid.uuid4().hex[:8]}",
            amount=round(amount, 2),
            currency='USD',
            timestamp=timestamp,
            transaction_type='wire',
            label='layering',
            memo=fake.bs()  # Business speak for layering
        )
        edges_involved.append((src, tgt))
    
    # Validate no cycles in the chain
    if not validate_no_cycles(G, edges_involved):
        raise ValueError("Cycle detected in layering chain - this should not happen")
    
    # Create crime record
    crime = InjectedCrime(
        crime_type='layering',
        nodes_involved=chain_nodes,
        edges_involved=edges_involved,
        metadata={
            'source_node': source_node,
            'dest_node': dest_node,
            'chain_length': config.chain_length,
            'initial_amount': config.initial_amount,
            'final_amount': round(amounts[-1], 2),
            'total_decay': round(1 - (amounts[-1] / config.initial_amount), 4),
            'amounts': amounts,
            'decays': decays
        }
    )
    
    return G, crime


def validate_no_cycles(G: nx.DiGraph, crime_edges: List[Tuple[int, int]]) -> bool:
    """
    Validate that injected crime edges don't create cycles.
    
    Args:
        G: NetworkX DiGraph
        crime_edges: List of edges that were added for crime
    
    Returns:
        True if no cycles exist involving crime edges
    """
    # Create subgraph with crime edges
    crime_subgraph = G.edge_subgraph(crime_edges)
    
    # Check for cycles
    try:
        nx.find_cycle(crime_subgraph)
        return False  # Cycle found
    except nx.NetworkXNoCycle:
        return True  # No cycles


def get_crime_labels(G: nx.DiGraph) -> Dict[Tuple[int, int], str]:
    """
    Extract crime labels from graph for training data.
    
    Args:
        G: NetworkX DiGraph with injected crimes
    
    Returns:
        Dictionary mapping edges (as tuples) to crime labels
    """
    labels = {}
    for u, v, data in G.edges(data=True):
        labels[(u, v)] = data.get('label', 'legitimate')
    return labels


def save_ground_truth(crime: InjectedCrime, filepath: Union[str, 'Path']) -> None:
    """
    Save InjectedCrime ground truth to JSON file.
    
    Args:
        crime: InjectedCrime dataclass instance
        filepath: Output file path
    """
    logger.info(f"Saving ground truth ({crime.crime_type}) to {filepath}")
    
    # Convert dataclass to dict, handling non-JSON-serializable types
    data = asdict(crime)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    logger.info(f"Ground truth saved: {len(crime.nodes_involved)} nodes, {len(crime.edges_involved)} edges")


def load_ground_truth(filepath: Union[str, 'Path']) -> Dict[str, Any]:
    """
    Load ground truth metadata from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Dictionary containing ground truth data
    """
    logger.info(f"Loading ground truth from {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Ground truth loaded: {data.get('crime_type', 'unknown')} crime")
    return data


__all__ = [
    'StructuringConfig',
    'LayeringConfig',
    'InjectedCrime',
    'inject_structuring',
    'inject_layering',
    'validate_no_cycles',
    'get_crime_labels',
    'save_ground_truth',
    'load_ground_truth'
]
