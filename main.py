"""
Main entry point for Green Financial Crime Agent.

The Panopticon Protocol: Zero-Failure Synthetic Financial Crime Simulator

Usage:
    python main.py generate --output-dir ./outputs
    python main.py generate --output-dir ./outputs --difficulty 8
    python main.py serve
"""
import argparse
import logging
import json
from pathlib import Path

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
from src.core.a2a_interface import app, set_graph, set_ground_truth, set_evidence

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(output_dir: Path, seed: int = 42, difficulty: int = 5) -> None:
    """
    Generate complete synthetic financial crime dataset WITH EVIDENCE.
    
    Creates a scale-free graph with entity and transaction attributes,
    injects structuring and layering crime patterns with configurable difficulty,
    generates evidence artifacts, and saves all outputs.
    
    Args:
        output_dir: Directory for output files
        seed: Random seed for reproducibility
        difficulty: Crime detection difficulty (1=trivial, 10=expert)
    """
    logger.info("=" * 60)
    logger.info("Starting ENHANCED synthetic financial crime data generation")
    logger.info(f"Difficulty Level: {difficulty}/10")
    logger.info("=" * 60)
    
    # Step 1: Generate baseline graph
    logger.info("Step 1: Generating scale-free baseline economy...")
    G = generate_scale_free_graph(n_nodes=1000, seed=seed)
    
    # Convert MultiDiGraph to DiGraph if needed (scale_free_graph returns MultiDiGraph)
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
    
    # Step 4: Inject structuring crime with difficulty and evidence
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
    
    # Step 5: Inject layering crime with difficulty and evidence
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
    logger.info(f"  - Total decay: {layering_crime.metadata['total_decay'] * 100:.2f}%")
    logger.info(f"  - Evidence artifacts: {len(layering_evidence)}")
    
    # Step 6: Collect all evidence artifacts
    logger.info("Step 6: Collecting evidence artifacts...")
    all_evidence = []
    all_evidence.extend(structuring_evidence)
    all_evidence.extend(layering_evidence)
    
    # Save evidence separately
    evidence_path = output_dir / "evidence_documents.json"
    with open(evidence_path, 'w') as f:
        json.dump(all_evidence, f, indent=2, default=str)
    logger.info(f"  - Evidence saved: {evidence_path}")
    logger.info(f"  - Total evidence documents: {len(all_evidence)}")
    
    # Step 7: Save outputs
    logger.info("Step 7: Saving outputs...")
    
    graph_path = output_dir / "final_graph.pkl"
    save_graph(G, graph_path)
    
    structuring_gt_path = output_dir / "structuring_gt.json"
    save_ground_truth(structuring_crime, structuring_gt_path)
    
    layering_gt_path = output_dir / "layering_gt.json"
    save_ground_truth(layering_crime, layering_gt_path)
    
    # Step 8: Prepare and save combined ground truth
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
    
    # Step 9: Load into A2A interface (for serve mode)
    logger.info("Step 9: Loading data into A2A interface...")
    set_graph(G)
    set_ground_truth(ground_truth)
    set_evidence(all_evidence)
    
    logger.info("=" * 60)
    logger.info("ENHANCED generation complete!")
    logger.info(f"  - Difficulty: {difficulty}/10")
    logger.info(f"  - Graph: {graph_path}")
    logger.info(f"  - Ground truth: {ground_truth_path}")
    logger.info(f"  - Evidence documents: {len(all_evidence)}")
    logger.info(f"  - Total nodes: {G.number_of_nodes()}")
    logger.info(f"  - Total edges: {G.number_of_edges()}")
    logger.info("=" * 60)


def start_server(host: str = "127.0.0.1", port: int = 5000, reload: bool = False) -> None:
    """
    Start A2A interface server.
    
    Args:
        host: Host to bind to (default: 127.0.0.1 for sidecar proxy)
        port: Port to listen on (default: 5000, proxied by sidecar on 8000)
        reload: Enable auto-reload for development
    """
    import uvicorn
    
    logger.info("=" * 60)
    logger.info("Starting Green Financial Crime Agent A2A Server")
    logger.info("=" * 60)
    logger.info(f"  - Internal Address: {host}:{port}")
    logger.info(f"  - External Access: Via Sidecar Proxy on :8000")
    logger.info(f"  - API Docs: http://{host}:{port}/docs")
    logger.info(f"  - Agent Manifest: http://{host}:{port}/agent.json")
    logger.info("=" * 60)
    
    uvicorn.run(
        "src.core.a2a_interface:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Green Financial Crime Agent - Synthetic AML Data Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate synthetic data (default difficulty 5)
    python main.py generate --output-dir ./outputs
    
    # Generate easy crimes (difficulty 3)
    python main.py generate --output-dir ./outputs/easy --difficulty 3
    
    # Generate expert-level crimes (difficulty 10)
    python main.py generate --output-dir ./outputs/expert --difficulty 10
    
    # Start A2A server (requires data to be generated first)
    python main.py serve
    
    # Start server with custom port
    python main.py serve --port 9000
    
    # Generate data and start server in one command (RECOMMENDED)
    python main.py serve --generate-on-startup --seed 42 --difficulty 5
    
    # Generate expert-level data and serve
    python main.py serve --generate-on-startup --difficulty 10
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Generate command
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate synthetic financial crime dataset"
    )
    gen_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs"),
        help="Output directory for generated files (default: ./outputs)"
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    gen_parser.add_argument(
        "--difficulty",
        type=int,
        default=5,
        choices=range(1, 11),
        metavar="1-10",
        help="Crime detection difficulty (1=trivial, 10=expert, default: 5)"
    )
    
    # Serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start A2A interface server"
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1 for sidecar proxy)"
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to (default: 5000, proxied by sidecar on 8000)"
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    serve_parser.add_argument(
        "--generate-on-startup",
        action="store_true",
        help="Generate synthetic data before starting server"
    )
    serve_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data generation (default: 42, requires --generate-on-startup)"
    )
    serve_parser.add_argument(
        "--difficulty",
        type=int,
        default=5,
        choices=range(1, 11),
        metavar="1-10",
        help="Crime difficulty (default: 5, requires --generate-on-startup)"
    )
    serve_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs"),
        help="Output directory for generated files (default: ./outputs)"
    )
    
    args = parser.parse_args()
    
    if args.command == "generate":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        generate_synthetic_data(
            args.output_dir, 
            seed=args.seed,
            difficulty=args.difficulty
        )
    elif args.command == "serve":
        # Generate data on startup if requested
        if args.generate_on_startup:
            logger.info("Generating data on startup...")
            args.output_dir.mkdir(parents=True, exist_ok=True)
            generate_synthetic_data(
                args.output_dir,
                seed=args.seed,
                difficulty=args.difficulty
            )
        start_server(host=args.host, port=args.port, reload=args.reload)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
