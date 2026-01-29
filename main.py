"""
Main entry point for Green Financial Crime Agent.

The Panopticon Protocol: Zero-Failure Synthetic Financial Crime Simulator

Usage:
    python main.py generate --output-dir ./outputs
    python main.py serve
"""
import argparse
import logging
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
from src.core.a2a_interface import app, set_graph, set_ground_truth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(output_dir: Path, seed: int = 42) -> None:
    """
    Generate complete synthetic financial crime dataset.
    
    Creates a scale-free graph with entity and transaction attributes,
    injects structuring and layering crime patterns, and saves all outputs.
    
    Args:
        output_dir: Directory for output files
        seed: Random seed for reproducibility
    """
    logger.info("=" * 60)
    logger.info("Starting synthetic financial crime data generation")
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
    logger.info("Step 2: Adding Faker-generated entity attributes...")
    G = add_entity_attributes(G, seed=seed)
    logger.info("  - Added names, addresses, SWIFT codes to all nodes")
    
    # Step 3: Add transaction attributes
    logger.info("Step 3: Adding transaction attributes to edges...")
    G = add_transaction_attributes(G, seed=seed)
    logger.info("  - Added amounts, timestamps, transaction types to all edges")
    
    # Step 4: Inject structuring crime
    logger.info("Step 4: Injecting structuring (smurfing) crime pattern...")
    mule_id = list(G.nodes())[0]
    structuring_config = StructuringConfig(mule_node=mule_id)
    G, structuring_crime = inject_structuring(G, config=structuring_config, seed=seed)
    logger.info(f"  - Mule node: {mule_id}")
    logger.info(f"  - Source nodes: {structuring_config.num_sources}")
    logger.info(f"  - Total amount: ${structuring_crime.metadata['total_amount']:,.2f}")
    
    # Step 5: Inject layering crime
    logger.info("Step 5: Injecting layering crime pattern...")
    source_node = list(G.nodes())[10]
    dest_node = list(G.nodes())[20]
    layering_config = LayeringConfig(chain_length=5)
    G, layering_crime = inject_layering(
        G, 
        config=layering_config, 
        source_node=source_node, 
        dest_node=dest_node, 
        seed=seed + 1  # Different seed for variety
    )
    logger.info(f"  - Chain length: {layering_config.chain_length}")
    logger.info(f"  - Initial amount: ${layering_crime.metadata['initial_amount']:,.2f}")
    logger.info(f"  - Final amount: ${layering_crime.metadata['final_amount']:,.2f}")
    logger.info(f"  - Total decay: {layering_crime.metadata['total_decay'] * 100:.2f}%")
    
    # Step 6: Save outputs
    logger.info("Step 6: Saving outputs...")
    
    graph_path = output_dir / "final_graph.pkl"
    save_graph(G, graph_path)
    
    structuring_gt_path = output_dir / "structuring_gt.json"
    save_ground_truth(structuring_crime, structuring_gt_path)
    
    layering_gt_path = output_dir / "layering_gt.json"
    save_ground_truth(layering_crime, layering_gt_path)
    
    logger.info("=" * 60)
    logger.info("Generation complete!")
    logger.info(f"  - Graph: {graph_path}")
    logger.info(f"  - Structuring ground truth: {structuring_gt_path}")
    logger.info(f"  - Layering ground truth: {layering_gt_path}")
    logger.info(f"  - Total nodes: {G.number_of_nodes()}")
    logger.info(f"  - Total edges: {G.number_of_edges()}")
    logger.info("=" * 60)


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """
    Start A2A interface server.
    
    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload for development
    """
    import uvicorn
    
    logger.info("=" * 60)
    logger.info("Starting Green Financial Crime Agent A2A Server")
    logger.info("=" * 60)
    logger.info(f"  - Host: {host}")
    logger.info(f"  - Port: {port}")
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
    # Generate synthetic data
    python main.py generate --output-dir ./outputs
    
    # Start A2A server
    python main.py serve
    
    # Start server with custom port
    python main.py serve --port 9000
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
    
    # Serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start A2A interface server"
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)"
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    if args.command == "generate":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        generate_synthetic_data(args.output_dir, seed=args.seed)
    elif args.command == "serve":
        start_server(host=args.host, port=args.port, reload=args.reload)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
