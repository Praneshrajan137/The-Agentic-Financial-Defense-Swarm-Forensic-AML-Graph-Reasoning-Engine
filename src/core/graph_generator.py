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
from typing import Optional, List, Union, Callable, Any
import random
import uuid
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
from faker import Faker
import numpy as np

logger = logging.getLogger(__name__)

# SDV import with graceful fallback
# We need to check if SDV is actually installed, not just if our module imports
try:
    import sdv
    from .sdv_models import get_transaction_synthesizer
    SDV_AVAILABLE = True
    logger.info("SDV successfully imported for correlated data generation")
except ImportError as e:
    SDV_AVAILABLE = False
    get_transaction_synthesizer = None  # type: ignore[assignment,no-redef]
    logger.warning(f"SDV not available, using random fallback: {e}")


# Country code to Faker locale mapping (CORRECTED)
COUNTRY_TO_LOCALE = {
    'US': 'en_US',
    'GB': 'en_GB',
    'IN': 'en_IN',
    'DE': 'de_DE',
    'FR': 'fr_FR',
    'IT': 'it_IT',
    'ES': 'es_ES',
    'JP': 'ja_JP',
    'CN': 'zh_CN',
    'BR': 'pt_BR',
    'CA': 'en_CA',
    'AU': 'en_AU',
    'NL': 'nl_NL',
    'CH': 'de_CH',
    'SE': 'sv_SE',
    'NO': 'nb_NO',  # Norwegian Bokmål (CORRECTED from 'no_NO')
    'DK': 'da_DK',  # Danish (CORRECTED from 'dk_DK')
    'PL': 'pl_PL',
    'RU': 'ru_RU',
    'TR': 'tr_TR',
}

# Supported countries for entity generation
SUPPORTED_COUNTRIES = ['US', 'GB', 'IN', 'DE', 'FR', 'IT', 'ES', 'JP']


def get_localized_faker(country_code: str) -> Faker:
    """
    Return a Faker instance locked to the specified country's locale.
    
    This ensures SWIFT codes, IBANs, and addresses are mathematically
    consistent with the country jurisdiction.
    
    Args:
        country_code: ISO 3166-1 alpha-2 country code
        
    Returns:
        Faker instance configured for that country's locale
    """
    locale = COUNTRY_TO_LOCALE.get(country_code, 'en_US')  # Fallback to US
    return Faker(locale)


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
    faker_instance=None,  # DEPRECATED - ignored for locale safety
    locales: Optional[List[str]] = None,  # DEPRECATED - ignored for locale safety
    seed: Optional[int] = None
) -> nx.DiGraph:
    """
    Add entity attributes with LOCALE-ALIGNED generation.
    
    CRITICAL FIX: We now generate country FIRST, then use a locale-locked
    Faker to ensure SWIFT codes and IBANs match the jurisdiction.
    
    Args:
        G: NetworkX DiGraph
        faker_instance: DEPRECATED (ignored for locale safety)
        locales: DEPRECATED (ignored for locale safety)
        seed: Random seed for reproducibility
    
    Returns:
        Graph with locale-consistent entity attributes
    """
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)
    
    entity_types = ['person', 'company', 'bank']
    entity_weights = [0.7, 0.25, 0.05]
    
    for node in G.nodes():
        # Step 1: Select country FIRST
        country = random.choice(SUPPORTED_COUNTRIES)
        
        # Step 2: Get locale-locked Faker for that country
        fake = get_localized_faker(country)
        
        # Step 3: Generate all attributes using THIS locale
        entity_type = random.choices(entity_types, weights=entity_weights)[0]
        G.nodes[node]['entity_type'] = entity_type
        G.nodes[node]['country'] = country  # Set BEFORE other attributes
        
        if entity_type == 'person':
            G.nodes[node]['name'] = fake.name()
        elif entity_type == 'company':
            G.nodes[node]['name'] = fake.company()
        else:  # bank
            G.nodes[node]['name'] = f"{fake.company()} Bank"
        
        # Generate locale-specific attributes
        G.nodes[node]['address'] = fake.address().replace('\n', ', ')
        
        # CRITICAL: SWIFT codes - use locale-specific with fallback
        # SWIFT format: 4 bank code + 2 country code + 2 location + optional 3 branch
        try:
            G.nodes[node]['swift'] = fake.swift()
        except AttributeError:
            # Fallback if locale doesn't support SWIFT
            # CORRECTED: Country code at positions 5-6, not at start
            G.nodes[node]['swift'] = f"XXXX{country}XX"
        
        # IBAN generation with fallback
        try:
            G.nodes[node]['iban'] = fake.iban()
        except AttributeError:
            # Fallback if locale doesn't support IBAN
            G.nodes[node]['iban'] = f"{country}{random.randint(10000000, 99999999)}"
        
        G.nodes[node]['risk_score'] = round(random.uniform(0, 1), 2)
        G.nodes[node]['verification_status'] = 'verified'
    
    logger.info(f"Locale-aligned entities: {G.number_of_nodes()} nodes across {len(SUPPORTED_COUNTRIES)} countries")
    return G


def add_transaction_attributes(
    G: nx.DiGraph,
    seed: Optional[int] = None,
    base_time: Optional[datetime] = None,
    use_sdv: bool = True
) -> nx.DiGraph:
    """
    Add transaction attributes to graph edges using SDV Gaussian Copula.
    
    This function uses SDV for statistically correlated data generation when available,
    with automatic fallback to random generation if SDV is not installed.
    
    SDV Correlation Rules:
    - High risk entities have larger transaction amounts
    - International transfers are systematically larger
    - Transaction types correlate with realistic spending patterns
    
    Args:
        G: NetworkX DiGraph or MultiDiGraph
        seed: Random seed for reproducibility
        base_time: Base timestamp for transactions (default: now)
        use_sdv: Whether to use SDV for correlated generation (default: True)
    
    Returns:
        Graph with transaction attributes added to edges
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if base_time is None:
        base_time = datetime.now()
    
    num_edges = G.number_of_edges()
    
    # Use SDV if available and requested
    if use_sdv and SDV_AVAILABLE and num_edges > 0:
        return _add_transaction_attributes_sdv(G, num_edges, base_time, seed)
    else:
        return _add_transaction_attributes_random(G, base_time)


def _add_transaction_attributes_sdv(
    G: nx.DiGraph,
    num_edges: int,
    base_time: datetime,
    seed: Optional[int] = None
) -> nx.DiGraph:
    """
    Add transaction attributes using SDV Gaussian Copula synthesizer.
    
    Generates statistically correlated transaction data where:
    - Amount correlates with risk_score
    - International transactions have higher amounts
    - Transaction types follow realistic distributions
    """
    # Get the trained synthesizer
    synthesizer = get_transaction_synthesizer()
    
    # Generate synthetic transaction data in batch with seed for reproducibility
    logger.info(f"Generating {num_edges} synthetic transactions via SDV Gaussian Copula...")
    # CRITICAL: Set seed BEFORE sampling for reproducibility
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    synthetic_tx = synthesizer.sample(num_rows=num_edges)
    
    # Convert to list of dicts for iteration
    tx_data = synthetic_tx.to_dict('records')
    
    # Assign to edges
    edge_idx = 0
    if isinstance(G, nx.MultiDiGraph):
        for u, v, key in G.edges(keys=True):
            tx = tx_data[edge_idx]
            
            G.edges[u, v, key]['transaction_id'] = f"txn_{uuid.uuid4().hex[:8]}"
            # Clamp amount to valid range [100, 50000]
            amount = max(100.0, min(float(tx['amount']), 50000.0))
            G.edges[u, v, key]['amount'] = amount
            G.edges[u, v, key]['risk_score'] = float(tx['risk_score'])
            G.edges[u, v, key]['is_international'] = bool(tx['is_international'])
            G.edges[u, v, key]['currency'] = 'USD'
            G.edges[u, v, key]['timestamp'] = base_time - timedelta(days=random.randint(0, 365))
            G.edges[u, v, key]['transaction_type'] = str(tx['transaction_type'])
            G.edges[u, v, key]['label'] = 'legitimate'
            G.edges[u, v, key]['memo'] = None
            
            edge_idx += 1
    else:
        for u, v in G.edges():
            tx = tx_data[edge_idx]
            
            G.edges[u, v]['transaction_id'] = f"txn_{uuid.uuid4().hex[:8]}"
            # Clamp amount to valid range [100, 50000]
            amount = max(100.0, min(float(tx['amount']), 50000.0))
            G.edges[u, v]['amount'] = amount
            G.edges[u, v]['risk_score'] = float(tx['risk_score'])
            G.edges[u, v]['is_international'] = bool(tx['is_international'])
            G.edges[u, v]['currency'] = 'USD'
            G.edges[u, v]['timestamp'] = base_time - timedelta(days=random.randint(0, 365))
            G.edges[u, v]['transaction_type'] = str(tx['transaction_type'])
            G.edges[u, v]['label'] = 'legitimate'
            G.edges[u, v]['memo'] = None
            
            edge_idx += 1
    
    logger.info(f"SDV transaction attributes assigned to {num_edges} edges")
    return G


def _add_transaction_attributes_random(
    G: nx.DiGraph,
    base_time: datetime
) -> nx.DiGraph:
    """
    Add transaction attributes using random generation (fallback method).
    
    This is the original implementation used when SDV is not available.
    """
    transaction_types = ['wire', 'ach', 'cash', 'internal']
    
    logger.info("Using random fallback for transaction attributes (SDV not available)")
    
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
    'load_graph',
    'get_localized_faker',
    'COUNTRY_TO_LOCALE',
    'SUPPORTED_COUNTRIES'
]
