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

from .evidence_generator import EvidenceGenerator

logger = logging.getLogger(__name__)


@dataclass
class StructuringConfig:
    """Configuration for structuring crime injection with difficulty."""
    num_sources: int = 20
    mule_node: Optional[int] = None
    min_amount: float = 9000.0
    max_amount: float = 9800.0
    time_window_hours: int = 48
    difficulty: int = 5  # 1-10 scale: 1=trivial, 10=expert


@dataclass
class LayeringConfig:
    """Configuration for layering crime injection with difficulty."""
    chain_length: int = 5
    min_decay: float = 0.02
    max_decay: float = 0.05
    initial_amount: float = 100000.0
    difficulty: int = 5  # 1-10 scale: 1=trivial, 10=expert


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
    seed: Optional[int] = None,
    generate_evidence: bool = True
) -> Tuple[nx.DiGraph, InjectedCrime]:
    """
    Inject structuring (smurfing) crime pattern with difficulty-based obfuscation.
    
    Pattern: Fan-in - multiple sources sending small amounts to single mule.
    
    Difficulty effects:
    - 1-3 (Trivial): All within 4 hours, similar amounts ($9,500-$9,700)
    - 4-6 (Medium): Spread over 48 hours, varied amounts ($9,000-$9,800)
    - 7-8 (Hard): Spread over 1 week, wider amounts ($7,500-$9,800), decoy transactions
    - 9-10 (Expert): Spread over 3 months, long gaps, mixed with legitimate activity
    
    Args:
        G: NetworkX DiGraph to inject crime into
        config: Structuring configuration parameters
        seed: Random seed for reproducibility
        generate_evidence: Whether to generate SAR and email artifacts (default: True)
    
    Returns:
        Tuple of (modified graph, crime record)
    """
    if config is None:
        config = StructuringConfig()
    
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)
    
    fake = Faker()
    difficulty = config.difficulty
    
    # Get the maximum node ID to create new unique nodes
    max_node = max(G.nodes()) if G.nodes() else 0
    
    # Select or use existing mule node
    if config.mule_node is not None:
        mule_id = config.mule_node
    else:
        # Select a random existing node as the mule
        mule_id = random.choice(list(G.nodes()))
    
    # DIFFICULTY-BASED TIME SPREADING
    if difficulty <= 3:
        # Trivial: All within 4 hours
        effective_time_window = 4
    elif difficulty <= 6:
        # Medium: Use configured window (48 hours default)
        effective_time_window = config.time_window_hours
    elif difficulty <= 8:
        # Hard: Spread over 1 week
        effective_time_window = 168  # 7 days
    else:
        # Expert: Spread over 3 months
        effective_time_window = 2160  # 90 days
    
    # DIFFICULTY-BASED AMOUNT RANGES
    if difficulty <= 3:
        # Trivial: Very similar amounts (easy to spot)
        min_amt = 9500.0
        max_amt = 9700.0
    elif difficulty <= 6:
        # Medium: Standard variance
        min_amt = config.min_amount
        max_amt = config.max_amount
    else:
        # Hard/Expert: Wider range, harder to detect
        min_amt = max(config.min_amount - 1500, 7500.0)
        max_amt = config.max_amount
    
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
        
        # Generate amount with difficulty-based variance
        amount = round(random.uniform(min_amt, max_amt), 2)
        amounts.append(amount)
        
        # Generate timestamp with difficulty-based spreading
        hours_offset = random.uniform(0, effective_time_window)
        
        # Expert mode: Add random "quiet periods" (long gaps)
        if difficulty >= 9 and i % 5 == 0:
            hours_offset += random.uniform(100, 500)
        
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
        
        # EXPERT MODE: Add decoy legitimate transactions
        if difficulty >= 8:
            # Add 2-4 legitimate transactions per smurf
            num_decoys = random.randint(2, 4)
            existing_nodes = list(G.nodes())
            for _ in range(num_decoys):
                decoy_target = random.choice(existing_nodes)
                if decoy_target != mule_id and decoy_target != source_id:
                    G.add_edge(
                        source_id,
                        decoy_target,
                        transaction_id=f"txn_{uuid.uuid4().hex[:8]}",
                        amount=round(random.uniform(50, 500), 2),
                        currency='USD',
                        timestamp=timestamp + timedelta(hours=random.uniform(-24, 24)),
                        transaction_type='ach',
                        label='legitimate',
                        memo=fake.sentence(nb_words=3)
                    )
    
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
            'effective_time_window_hours': effective_time_window,
            'amounts': amounts,
            'difficulty': difficulty,
            'amount_range': [min_amt, max_amt]
        }
    )
    
    # Generate evidence artifacts
    evidence_artifacts = []
    
    if generate_evidence:
        evidence_gen = EvidenceGenerator(seed=seed)
        
        # Get mule name from graph
        mule_name = G.nodes[mule_id].get('name', f'Entity {mule_id}')
        
        # Generate SAR narrative
        sar = evidence_gen.generate_sar_narrative(
            subject_id=str(mule_id),
            subject_name=mule_name,
            crime_type='structuring',
            transaction_count=config.num_sources,
            total_amount=round(sum(amounts), 2),
            time_window_hours=config.time_window_hours
        )
        evidence_artifacts.append(sar)
        
        # Generate internal emails from branch managers (3 emails)
        for i in range(min(3, len(source_nodes))):
            source_id = source_nodes[i]
            source_name = G.nodes[source_id].get('name', f'Smurf {i}')
            
            email = evidence_gen.generate_internal_email(
                subject_id=str(source_id),
                subject_name=source_name,
                suspicious_behavior=f"Customer made a ${amounts[i]:,.2f} cash deposit and asked about CTR limits"
            )
            evidence_artifacts.append(email)
        
        logger.info(f"Generated {len(evidence_artifacts)} evidence artifacts for structuring crime")
    
    # Add evidence to crime metadata
    crime.metadata['evidence_artifacts'] = evidence_artifacts
    crime.metadata['evidence_count'] = len(evidence_artifacts)
    
    return G, crime


def inject_layering(
    G: nx.DiGraph,
    config: Optional[LayeringConfig] = None,
    seed: Optional[int] = None,
    source_node: Optional[int] = None,
    dest_node: Optional[int] = None,
    generate_evidence: bool = True
) -> Tuple[nx.DiGraph, InjectedCrime]:
    """
    Inject layering crime pattern with difficulty-based obfuscation.
    
    Pattern: Chain transfers with decay to obscure money trail.
    
    Difficulty effects:
    - 1-3 (Trivial): Short chain (3 hops), obvious decay (5%), rapid transfers
    - 4-6 (Medium): Medium chain (5-7 hops), realistic decay (2-5%)
    - 7-8 (Hard): Long chain (8-10 hops), varied decay, mixed with legitimate
    - 9-10 (Expert): Very long chain (15+ hops), minimal decay (1-2%), long time gaps
    
    Args:
        G: NetworkX DiGraph to inject crime into
        config: Layering configuration parameters
        seed: Random seed for reproducibility
        source_node: Optional starting node (uses random existing node if None)
        dest_node: Optional destination node (uses random existing node if None)
        generate_evidence: Whether to generate SAR and email artifacts (default: True)
    
    Returns:
        Tuple of (modified graph, crime record)
    """
    if config is None:
        config = LayeringConfig()
    
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)
    
    fake = Faker()
    difficulty = config.difficulty
    
    # DIFFICULTY-BASED CHAIN LENGTH
    if difficulty <= 3:
        # Trivial: Short chain, easy to trace
        effective_chain_length = 3
    elif difficulty <= 6:
        # Medium: Use configured length
        effective_chain_length = config.chain_length
    elif difficulty <= 8:
        # Hard: Longer chain
        effective_chain_length = random.randint(8, 10)
    else:
        # Expert: Very long chain
        effective_chain_length = random.randint(15, 20)
    
    # DIFFICULTY-BASED DECAY RATES
    if difficulty <= 3:
        # Trivial: Obvious decay (easy to spot)
        min_decay = 0.04
        max_decay = 0.06
    elif difficulty <= 6:
        # Medium: Standard decay
        min_decay = config.min_decay
        max_decay = config.max_decay
    else:
        # Hard/Expert: Minimal decay (harder to detect)
        min_decay = 0.01
        max_decay = 0.02
    
    # DIFFICULTY-BASED TIME INTERVALS
    if difficulty <= 3:
        # Trivial: Very rapid (easy to spot velocity)
        hop_interval_minutes = 15
    elif difficulty <= 6:
        # Medium: 30 minutes between hops
        hop_interval_minutes = 30
    elif difficulty <= 8:
        # Hard: Hours between hops
        hop_interval_minutes = random.randint(60, 240)
    else:
        # Expert: Days between hops
        hop_interval_minutes = random.randint(720, 2880)  # 12-48 hours
    
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
    for i in range(effective_chain_length):
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
        
        # EXPERT MODE: Add decoy legitimate transactions
        if difficulty >= 8:
            num_decoys = random.randint(1, 3)
            for _ in range(num_decoys):
                decoy_target = random.choice(existing_nodes)
                if decoy_target != new_node:
                    G.add_edge(
                        new_node,
                        decoy_target,
                        transaction_id=f"txn_{uuid.uuid4().hex[:8]}",
                        amount=round(random.uniform(100, 5000), 2),
                        currency='USD',
                        timestamp=datetime.now() + timedelta(hours=random.uniform(-24, 24)),
                        transaction_type='ach',
                        label='legitimate',
                        memo=fake.sentence(nb_words=3)
                    )
    
    # Add destination to chain
    chain_nodes.append(dest_node)
    
    # Build edges with decay
    edges_involved = []
    amounts = []
    decays = []
    amount = config.initial_amount
    base_time = datetime.now()
    
    for i in range(len(chain_nodes) - 1):
        src = chain_nodes[i]
        tgt = chain_nodes[i + 1]
        
        # Apply decay
        decay = random.uniform(min_decay, max_decay)
        decays.append(decay)
        
        if i > 0:  # Don't decay the first transfer
            amount = amount * (1 - decay)
        
        amounts.append(round(amount, 2))
        
        # Generate timestamp with difficulty-based velocity
        # Expert mode: Add occasional long gaps
        interval = hop_interval_minutes
        if difficulty >= 9 and i % 4 == 0:
            interval += random.randint(1440, 4320)  # Add 1-3 days gap
        
        timestamp = base_time + timedelta(minutes=interval * i)
        
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
            'effective_chain_length': effective_chain_length,
            'initial_amount': config.initial_amount,
            'final_amount': round(amounts[-1], 2),
            'total_decay': round(1 - (amounts[-1] / config.initial_amount), 4),
            'amounts': amounts,
            'decays': decays,
            'difficulty': difficulty,
            'decay_range': [min_decay, max_decay]
        }
    )
    
    # Generate evidence artifacts
    evidence_artifacts = []
    
    if generate_evidence:
        evidence_gen = EvidenceGenerator(seed=seed)
        
        # Get source name from graph
        source_name = G.nodes[source_node].get('name', f'Entity {source_node}')
        
        # Generate SAR narrative
        sar = evidence_gen.generate_sar_narrative(
            subject_id=str(source_node),
            subject_name=source_name,
            crime_type='layering',
            transaction_count=len(chain_nodes) - 1,
            total_amount=config.initial_amount,
            time_window_hours=24
        )
        evidence_artifacts.append(sar)
        
        # Generate conflicting evidence (GOLD MEDAL FEATURE)
        # This tests hallucination resistance
        if len(chain_nodes) > 1:
            first_amount = amounts[0] if amounts else config.initial_amount
            conflicts = evidence_gen.generate_conflicting_evidence(
                subject_id=str(chain_nodes[1]),  # First intermediate node
                actual_amount=first_amount,
                graph_amount=first_amount  # They match, but email is wrong
            )
            evidence_artifacts.extend(conflicts)
        
        logger.info(f"Generated {len(evidence_artifacts)} evidence artifacts for layering crime")
    
    # Add evidence to crime metadata
    crime.metadata['evidence_artifacts'] = evidence_artifacts
    crime.metadata['evidence_count'] = len(evidence_artifacts)
    
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
