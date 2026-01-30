#!/usr/bin/env python3
"""
Deterministic Server Initialization Script
==========================================
Generates synthetic data and starts the A2A server with deterministic configuration.

This script ensures reproducible startup by:
1. Setting all random seeds before any data generation
2. Generating synthetic financial crime data with fixed configuration
3. Loading data into the A2A interface
4. Starting the server

Usage:
    python scripts/init_server.py --seed 42 --difficulty 5 --port 8000

Environment Variables (override CLI args):
    SEED: Random seed (default: 42)
    DIFFICULTY: Crime difficulty 1-10 (default: 5)
    PORT: Server port (default: 8000)
    HOST: Server host (default: 0.0.0.0)
"""

import argparse
import logging
import os
import sys
import random
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_all_seeds(seed: int) -> None:
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: The seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Set Faker seed
    try:
        from faker import Faker
        Faker.seed(seed)
    except ImportError:
        pass
    
    # Set environment variable for any subprocess
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"All random seeds set to: {seed}")


def generate_and_load_data(
    output_dir: Path,
    seed: int = 42,
    difficulty: int = 5
) -> bool:
    """
    Generate synthetic data and load into A2A interface.
    
    Args:
        output_dir: Directory for output files
        seed: Random seed for reproducibility
        difficulty: Crime detection difficulty (1-10)
        
    Returns:
        True if successful
    """
    import json
    import networkx as nx
    
    from src.core.graph_generator import (
        generate_scale_free_graph,
        add_entity_attributes,
        add_transaction_attributes,
        save_graph
    )
    from src.core.crime_injector import (
        inject_structuring,
        inject_layering,
        StructuringConfig,
        LayeringConfig,
        save_ground_truth
    )
    from src.core.a2a_interface import set_graph, set_ground_truth, set_evidence
    
    logger.info("=" * 60)
    logger.info("DETERMINISTIC DATA GENERATION")
    logger.info(f"Seed: {seed}")
    logger.info(f"Difficulty: {difficulty}")
    logger.info("=" * 60)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate baseline graph
    logger.info("Step 1: Generating scale-free baseline economy...")
    G = generate_scale_free_graph(n_nodes=1000, seed=seed)
    
    # Convert MultiDiGraph to DiGraph if needed
    if isinstance(G, nx.MultiDiGraph):
        G = nx.DiGraph(G)
        logger.info("  - Converted MultiDiGraph to DiGraph")
    
    logger.info(f"  - Generated {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Step 2: Add entity attributes
    logger.info("Step 2: Adding locale-aligned entity attributes...")
    G = add_entity_attributes(G, seed=seed)
    logger.info("  - Added names, addresses, SWIFT/IBAN codes to all nodes")
    
    # Step 3: Add transaction attributes
    logger.info("Step 3: Adding SDV-correlated transaction attributes...")
    G = add_transaction_attributes(G, seed=seed)
    logger.info("  - Added amounts, timestamps, transaction types to all edges")
    
    # Step 4: Inject structuring crime
    logger.info(f"Step 4: Injecting structuring (difficulty {difficulty})...")
    mule_id = list(G.nodes())[0]
    structuring_config = StructuringConfig(
        mule_node=mule_id,
        difficulty=difficulty
    )
    G, structuring_crime = inject_structuring(
        G, 
        config=structuring_config, 
        seed=seed,
        generate_evidence=True
    )
    structuring_evidence = structuring_crime.metadata.get('evidence_artifacts', [])
    logger.info(f"  - Mule node: {mule_id}")
    logger.info(f"  - Source nodes: {structuring_config.num_sources}")
    logger.info(f"  - Total amount: ${structuring_crime.metadata['total_amount']:,.2f}")
    logger.info(f"  - Evidence artifacts: {len(structuring_evidence)}")
    
    # Step 5: Inject layering crime
    logger.info(f"Step 5: Injecting layering (difficulty {difficulty})...")
    source_node = list(G.nodes())[10]
    dest_node = list(G.nodes())[20]
    layering_config = LayeringConfig(
        chain_length=5,
        difficulty=difficulty
    )
    G, layering_crime = inject_layering(
        G, 
        config=layering_config, 
        source_node=source_node, 
        dest_node=dest_node, 
        seed=seed + 1,
        generate_evidence=True
    )
    layering_evidence = layering_crime.metadata.get('evidence_artifacts', [])
    logger.info(f"  - Chain length: {layering_crime.metadata.get('effective_chain_length', layering_config.chain_length)}")
    logger.info(f"  - Initial amount: ${layering_crime.metadata['initial_amount']:,.2f}")
    logger.info(f"  - Final amount: ${layering_crime.metadata['final_amount']:,.2f}")
    logger.info(f"  - Evidence artifacts: {len(layering_evidence)}")
    
    # Step 6: Collect all evidence
    logger.info("Step 6: Collecting evidence artifacts...")
    all_evidence = structuring_evidence + layering_evidence
    
    # Save evidence
    evidence_path = output_dir / "evidence_documents.json"
    with open(evidence_path, 'w') as f:
        json.dump(all_evidence, f, indent=2, default=str)
    logger.info(f"  - Evidence saved: {evidence_path}")
    
    # Step 7: Save outputs
    logger.info("Step 7: Saving outputs...")
    
    graph_path = output_dir / "final_graph.pkl"
    save_graph(G, graph_path)
    
    structuring_gt_path = output_dir / "structuring_gt.json"
    save_ground_truth(structuring_crime, structuring_gt_path)
    
    layering_gt_path = output_dir / "layering_gt.json"
    save_ground_truth(layering_crime, layering_gt_path)
    
    # Step 8: Prepare combined ground truth
    logger.info("Step 8: Preparing combined ground truth...")
    ground_truth = {
        'crimes': [
            {
                'crime_type': 'structuring',
                'nodes_involved': structuring_crime.nodes_involved,
                'edges_involved': structuring_crime.edges_involved,
                'metadata': {k: v for k, v in structuring_crime.metadata.items() if k != 'evidence_artifacts'}
            },
            {
                'crime_type': 'layering',
                'nodes_involved': layering_crime.nodes_involved,
                'edges_involved': layering_crime.edges_involved,
                'metadata': {k: v for k, v in layering_crime.metadata.items() if k != 'evidence_artifacts'}
            }
        ],
        'difficulty': difficulty,
        'total_evidence_artifacts': len(all_evidence)
    }
    
    ground_truth_path = output_dir / "ground_truth.json"
    with open(ground_truth_path, 'w') as f:
        json.dump(ground_truth, f, indent=2, default=str)
    
    # Step 9: Load into A2A interface
    logger.info("Step 9: Loading data into A2A interface...")
    set_graph(G)
    set_ground_truth(ground_truth)
    set_evidence(all_evidence)
    
    logger.info("=" * 60)
    logger.info("DATA GENERATION COMPLETE")
    logger.info(f"  - Seed: {seed}")
    logger.info(f"  - Difficulty: {difficulty}")
    logger.info(f"  - Graph: {graph_path}")
    logger.info(f"  - Ground truth: {ground_truth_path}")
    logger.info(f"  - Evidence documents: {len(all_evidence)}")
    logger.info(f"  - Total nodes: {G.number_of_nodes()}")
    logger.info(f"  - Total edges: {G.number_of_edges()}")
    logger.info("=" * 60)
    
    return True


def start_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Start the A2A server.
    
    Args:
        host: Host to bind to
        port: Port to listen on
    """
    import uvicorn
    
    logger.info("=" * 60)
    logger.info("STARTING A2A SERVER")
    logger.info(f"  - Host: {host}")
    logger.info(f"  - Port: {port}")
    logger.info(f"  - API Docs: http://{host}:{port}/docs")
    logger.info(f"  - Agent Manifest: http://{host}:{port}/agent.json")
    logger.info("=" * 60)
    
    uvicorn.run(
        "src.core.a2a_interface:app",
        host=host,
        port=port,
        log_level="info"
    )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Initialize and start the Green Financial Crime Agent with deterministic configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with default settings (seed=42, difficulty=5)
    python scripts/init_server.py
    
    # Start with custom configuration
    python scripts/init_server.py --seed 123 --difficulty 8 --port 9000
    
    # Use environment variables
    SEED=42 DIFFICULTY=5 PORT=8000 python scripts/init_server.py
        """
    )
    
    # Get defaults from environment variables
    default_seed = int(os.environ.get('SEED', 42))
    default_difficulty = int(os.environ.get('DIFFICULTY', 5))
    default_port = int(os.environ.get('PORT', 8000))
    default_host = os.environ.get('HOST', '0.0.0.0')
    
    parser.add_argument(
        '--seed',
        type=int,
        default=default_seed,
        help=f'Random seed for reproducibility (default: {default_seed})'
    )
    parser.add_argument(
        '--difficulty',
        type=int,
        default=default_difficulty,
        choices=range(1, 11),
        metavar='1-10',
        help=f'Crime detection difficulty (default: {default_difficulty})'
    )
    parser.add_argument(
        '--host',
        type=str,
        default=default_host,
        help=f'Host to bind to (default: {default_host})'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=default_port,
        help=f'Port to listen on (default: {default_port})'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PROJECT_ROOT / "outputs",
        help='Output directory for generated files'
    )
    parser.add_argument(
        '--generate-only',
        action='store_true',
        help='Only generate data, do not start server'
    )
    
    args = parser.parse_args()
    
    # Set all seeds first
    set_all_seeds(args.seed)
    
    # Generate and load data
    success = generate_and_load_data(
        output_dir=args.output_dir,
        seed=args.seed,
        difficulty=args.difficulty
    )
    
    if not success:
        logger.error("Data generation failed!")
        sys.exit(1)
    
    # Start server unless generate-only
    if not args.generate_only:
        start_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
