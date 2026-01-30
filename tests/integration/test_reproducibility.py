"""
Reproducibility Verification Tests
==================================
Tests that verify deterministic behavior when using the same seed.

These tests ensure that:
1. Same seed produces identical graphs (nodes, edges, attributes)
2. Same seed produces identical crime patterns (structuring, layering)
3. Same seed produces identical evidence artifacts
4. Different seeds produce different outputs
"""

import pytest
import json
import hashlib
import pickle
from datetime import datetime
from typing import Any, Dict, List

import networkx as nx


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def seed():
    """Fixed seed for reproducibility tests."""
    return 42


@pytest.fixture
def alternate_seed():
    """Different seed for non-reproducibility tests."""
    return 123


@pytest.fixture
def difficulty():
    """Fixed difficulty for tests."""
    return 5


# =============================================================================
# Helper Functions
# =============================================================================

def graph_hash(G: nx.DiGraph) -> str:
    """
    Compute a hash of the graph structure and attributes.
    
    Args:
        G: NetworkX graph
        
    Returns:
        SHA256 hash of graph content
    """
    # Collect node data
    node_data = []
    for node in sorted(G.nodes()):
        attrs = dict(G.nodes[node])
        # Remove timestamp fields that may vary
        attrs.pop('timestamp', None)
        attrs.pop('created_at', None)
        attrs.pop('last_activity', None)
        node_data.append((node, sorted(attrs.items())))
    
    # Collect edge data
    edge_data = []
    for u, v in sorted(G.edges()):
        attrs = dict(G.edges[u, v])
        # Remove timestamp fields
        attrs.pop('timestamp', None)
        edge_data.append((u, v, sorted(attrs.items())))
    
    # Create hash
    content = json.dumps({
        'nodes': node_data,
        'edges': edge_data
    }, sort_keys=True, default=str)
    
    return hashlib.sha256(content.encode()).hexdigest()


def crime_hash(crime: Dict[str, Any]) -> str:
    """
    Compute a hash of a crime record.
    
    Args:
        crime: Crime dictionary
        
    Returns:
        SHA256 hash of crime content
    """
    # Remove evidence artifacts and timestamps
    clean_crime = {
        'crime_type': crime.get('crime_type'),
        'nodes_involved': sorted(crime.get('nodes_involved', [])),
        'edges_involved': sorted([tuple(e) for e in crime.get('edges_involved', [])]),
        'metadata': {
            k: v for k, v in crime.get('metadata', {}).items()
            if k not in ['evidence_artifacts', 'timestamp']
        }
    }
    
    content = json.dumps(clean_crime, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()


# =============================================================================
# Graph Generation Reproducibility
# =============================================================================

@pytest.mark.integration
class TestGraphReproducibility:
    """Tests for graph generation reproducibility."""
    
    def test_same_seed_produces_identical_graphs(self, seed):
        """Test that same seed produces identical graph structure."""
        from src.core.graph_generator import generate_scale_free_graph
        
        # Generate two graphs with same seed
        G1 = generate_scale_free_graph(n_nodes=100, seed=seed)
        G2 = generate_scale_free_graph(n_nodes=100, seed=seed)
        
        # Verify identical structure
        assert G1.number_of_nodes() == G2.number_of_nodes()
        assert G1.number_of_edges() == G2.number_of_edges()
        assert set(G1.nodes()) == set(G2.nodes())
        assert set(G1.edges()) == set(G2.edges())
    
    def test_same_seed_produces_identical_entity_attributes(self, seed):
        """Test that same seed produces identical entity attributes."""
        from src.core.graph_generator import (
            generate_scale_free_graph,
            add_entity_attributes
        )
        
        # Generate and add attributes with same seed
        G1 = generate_scale_free_graph(n_nodes=50, seed=seed)
        G1 = add_entity_attributes(G1, seed=seed)
        
        G2 = generate_scale_free_graph(n_nodes=50, seed=seed)
        G2 = add_entity_attributes(G2, seed=seed)
        
        # Convert to DiGraph if needed
        if isinstance(G1, nx.MultiDiGraph):
            G1 = nx.DiGraph(G1)
        if isinstance(G2, nx.MultiDiGraph):
            G2 = nx.DiGraph(G2)
        
        # Verify identical node attributes
        for node in G1.nodes():
            assert node in G2.nodes()
            attrs1 = G1.nodes[node]
            attrs2 = G2.nodes[node]
            
            # Check key attributes
            assert attrs1.get('entity_type') == attrs2.get('entity_type')
            assert attrs1.get('name') == attrs2.get('name')
            assert attrs1.get('country') == attrs2.get('country')
    
    def test_different_seeds_produce_different_graphs(self, seed, alternate_seed):
        """Test that different seeds produce different graphs."""
        from src.core.graph_generator import (
            generate_scale_free_graph,
            add_entity_attributes
        )
        
        # Generate with different seeds
        G1 = generate_scale_free_graph(n_nodes=50, seed=seed)
        G1 = add_entity_attributes(G1, seed=seed)
        
        G2 = generate_scale_free_graph(n_nodes=50, seed=alternate_seed)
        G2 = add_entity_attributes(G2, seed=alternate_seed)
        
        # Convert to DiGraph if needed
        if isinstance(G1, nx.MultiDiGraph):
            G1 = nx.DiGraph(G1)
        if isinstance(G2, nx.MultiDiGraph):
            G2 = nx.DiGraph(G2)
        
        # Compute hashes
        hash1 = graph_hash(G1)
        hash2 = graph_hash(G2)
        
        # Should be different
        assert hash1 != hash2, "Different seeds should produce different graphs"
    
    def test_graph_hash_determinism(self, seed):
        """Test that graph hashing is deterministic."""
        from src.core.graph_generator import (
            generate_scale_free_graph,
            add_entity_attributes
        )
        
        # Generate same graph twice
        G1 = generate_scale_free_graph(n_nodes=50, seed=seed)
        G1 = add_entity_attributes(G1, seed=seed)
        
        G2 = generate_scale_free_graph(n_nodes=50, seed=seed)
        G2 = add_entity_attributes(G2, seed=seed)
        
        # Convert to DiGraph if needed
        if isinstance(G1, nx.MultiDiGraph):
            G1 = nx.DiGraph(G1)
        if isinstance(G2, nx.MultiDiGraph):
            G2 = nx.DiGraph(G2)
        
        # Hashes should be identical
        hash1 = graph_hash(G1)
        hash2 = graph_hash(G2)
        
        assert hash1 == hash2, "Same seed should produce identical graph hash"


# =============================================================================
# Crime Injection Reproducibility
# =============================================================================

@pytest.mark.integration
class TestCrimeReproducibility:
    """Tests for crime injection reproducibility."""
    
    def test_same_seed_produces_identical_structuring(self, seed, difficulty):
        """Test that same seed produces identical structuring crime."""
        from src.core.graph_generator import (
            generate_scale_free_graph,
            add_entity_attributes,
            add_transaction_attributes
        )
        from src.core.crime_injector import inject_structuring, StructuringConfig
        
        # First run
        G1 = generate_scale_free_graph(n_nodes=100, seed=seed)
        if isinstance(G1, nx.MultiDiGraph):
            G1 = nx.DiGraph(G1)
        G1 = add_entity_attributes(G1, seed=seed)
        G1 = add_transaction_attributes(G1, seed=seed)
        
        mule_id = list(G1.nodes())[0]
        config = StructuringConfig(mule_node=mule_id, difficulty=difficulty)
        G1, crime1 = inject_structuring(G1, config=config, seed=seed, generate_evidence=False)
        
        # Second run
        G2 = generate_scale_free_graph(n_nodes=100, seed=seed)
        if isinstance(G2, nx.MultiDiGraph):
            G2 = nx.DiGraph(G2)
        G2 = add_entity_attributes(G2, seed=seed)
        G2 = add_transaction_attributes(G2, seed=seed)
        
        config2 = StructuringConfig(mule_node=mule_id, difficulty=difficulty)
        G2, crime2 = inject_structuring(G2, config=config2, seed=seed, generate_evidence=False)
        
        # Verify identical crimes
        assert crime1.crime_type == crime2.crime_type
        assert crime1.nodes_involved == crime2.nodes_involved
        assert crime1.edges_involved == crime2.edges_involved
        assert crime1.metadata['total_amount'] == crime2.metadata['total_amount']
        assert crime1.metadata['amounts'] == crime2.metadata['amounts']
    
    def test_same_seed_produces_identical_layering(self, seed, difficulty):
        """Test that same seed produces identical layering crime."""
        from src.core.graph_generator import (
            generate_scale_free_graph,
            add_entity_attributes,
            add_transaction_attributes
        )
        from src.core.crime_injector import inject_layering, LayeringConfig
        
        # First run
        G1 = generate_scale_free_graph(n_nodes=100, seed=seed)
        if isinstance(G1, nx.MultiDiGraph):
            G1 = nx.DiGraph(G1)
        G1 = add_entity_attributes(G1, seed=seed)
        G1 = add_transaction_attributes(G1, seed=seed)
        
        source = list(G1.nodes())[10]
        dest = list(G1.nodes())[20]
        config = LayeringConfig(chain_length=5, difficulty=difficulty)
        G1, crime1 = inject_layering(G1, config=config, source_node=source, dest_node=dest, seed=seed, generate_evidence=False)
        
        # Second run
        G2 = generate_scale_free_graph(n_nodes=100, seed=seed)
        if isinstance(G2, nx.MultiDiGraph):
            G2 = nx.DiGraph(G2)
        G2 = add_entity_attributes(G2, seed=seed)
        G2 = add_transaction_attributes(G2, seed=seed)
        
        config2 = LayeringConfig(chain_length=5, difficulty=difficulty)
        G2, crime2 = inject_layering(G2, config=config2, source_node=source, dest_node=dest, seed=seed, generate_evidence=False)
        
        # Verify identical crimes
        assert crime1.crime_type == crime2.crime_type
        assert crime1.nodes_involved == crime2.nodes_involved
        assert crime1.edges_involved == crime2.edges_involved
        assert crime1.metadata['initial_amount'] == crime2.metadata['initial_amount']
        assert crime1.metadata['final_amount'] == crime2.metadata['final_amount']
        assert crime1.metadata['amounts'] == crime2.metadata['amounts']
    
    def test_different_seeds_produce_different_crimes(self, seed, alternate_seed, difficulty):
        """Test that different seeds produce different crime patterns."""
        from src.core.graph_generator import generate_scale_free_graph
        from src.core.crime_injector import inject_structuring, StructuringConfig
        
        # First run
        G1 = generate_scale_free_graph(n_nodes=100, seed=seed)
        if isinstance(G1, nx.MultiDiGraph):
            G1 = nx.DiGraph(G1)
        mule_id = list(G1.nodes())[0]
        config1 = StructuringConfig(mule_node=mule_id, difficulty=difficulty)
        G1, crime1 = inject_structuring(G1, config=config1, seed=seed, generate_evidence=False)
        
        # Second run with different seed
        G2 = generate_scale_free_graph(n_nodes=100, seed=alternate_seed)
        if isinstance(G2, nx.MultiDiGraph):
            G2 = nx.DiGraph(G2)
        mule_id2 = list(G2.nodes())[0]
        config2 = StructuringConfig(mule_node=mule_id2, difficulty=difficulty)
        G2, crime2 = inject_structuring(G2, config=config2, seed=alternate_seed, generate_evidence=False)
        
        # Amounts should be different (random)
        assert crime1.metadata['amounts'] != crime2.metadata['amounts']


# =============================================================================
# Evidence Generation Reproducibility
# =============================================================================

@pytest.mark.integration
class TestEvidenceReproducibility:
    """Tests for evidence generation reproducibility."""
    
    def test_same_seed_produces_identical_evidence(self, seed):
        """Test that same seed produces identical evidence artifacts."""
        from src.core.evidence_generator import EvidenceGenerator
        
        # First run
        gen1 = EvidenceGenerator(seed=seed)
        sar1 = gen1.generate_sar_narrative(
            subject_id="42",
            subject_name="Test Subject",
            crime_type="structuring",
            transaction_count=20,
            total_amount=180000.0,
            time_window_hours=48
        )
        
        # Second run
        gen2 = EvidenceGenerator(seed=seed)
        sar2 = gen2.generate_sar_narrative(
            subject_id="42",
            subject_name="Test Subject",
            crime_type="structuring",
            transaction_count=20,
            total_amount=180000.0,
            time_window_hours=48
        )
        
        # Key fields should be identical
        assert sar1['document_type'] == sar2['document_type']
        assert sar1['subject_id'] == sar2['subject_id']
        assert sar1['crime_type'] == sar2['crime_type']
        # File numbers should be the same due to seed
        assert sar1['file_number'] == sar2['file_number']
    
    def test_same_seed_produces_identical_emails(self, seed):
        """Test that same seed produces identical email evidence."""
        from src.core.evidence_generator import EvidenceGenerator
        
        # First run
        gen1 = EvidenceGenerator(seed=seed)
        email1 = gen1.generate_internal_email(
            subject_id="42",
            subject_name="Test Subject",
            suspicious_behavior="Made multiple cash deposits"
        )
        
        # Second run
        gen2 = EvidenceGenerator(seed=seed)
        email2 = gen2.generate_internal_email(
            subject_id="42",
            subject_name="Test Subject",
            suspicious_behavior="Made multiple cash deposits"
        )
        
        # Key fields should be identical
        assert email1['document_type'] == email2['document_type']
        assert email1['subject_id'] == email2['subject_id']
    
    def test_conflicting_evidence_reproducibility(self, seed):
        """Test that conflicting evidence generation is reproducible."""
        from src.core.evidence_generator import EvidenceGenerator
        
        # First run
        gen1 = EvidenceGenerator(seed=seed)
        conflicts1 = gen1.generate_conflicting_evidence(
            subject_id="42",
            actual_amount=9500.0,
            graph_amount=9500.0
        )
        
        # Second run
        gen2 = EvidenceGenerator(seed=seed)
        conflicts2 = gen2.generate_conflicting_evidence(
            subject_id="42",
            actual_amount=9500.0,
            graph_amount=9500.0
        )
        
        # Should produce same number of documents
        assert len(conflicts1) == len(conflicts2)
        
        # Document types should match
        for doc1, doc2 in zip(conflicts1, conflicts2):
            assert doc1['document_type'] == doc2['document_type']


# =============================================================================
# Full Pipeline Reproducibility
# =============================================================================

@pytest.mark.slow
@pytest.mark.integration
class TestFullPipelineReproducibility:
    """End-to-end reproducibility tests."""
    
    def test_full_pipeline_reproducibility(self, seed, difficulty, tmp_path):
        """Test that the full pipeline produces identical results with same seed."""
        from src.core.graph_generator import (
            generate_scale_free_graph,
            add_entity_attributes,
            add_transaction_attributes
        )
        from src.core.crime_injector import (
            inject_structuring,
            inject_layering,
            StructuringConfig,
            LayeringConfig
        )
        
        results = []
        
        for _ in range(2):
            # Generate graph
            G = generate_scale_free_graph(n_nodes=100, seed=seed)
            if isinstance(G, nx.MultiDiGraph):
                G = nx.DiGraph(G)
            G = add_entity_attributes(G, seed=seed)
            G = add_transaction_attributes(G, seed=seed)
            
            # Inject crimes
            mule_id = list(G.nodes())[0]
            struct_config = StructuringConfig(mule_node=mule_id, difficulty=difficulty)
            G, struct_crime = inject_structuring(G, config=struct_config, seed=seed, generate_evidence=True)
            
            source = list(G.nodes())[10]
            dest = list(G.nodes())[20]
            layer_config = LayeringConfig(chain_length=5, difficulty=difficulty)
            G, layer_crime = inject_layering(G, config=layer_config, source_node=source, dest_node=dest, seed=seed+1, generate_evidence=True)
            
            # Collect results
            results.append({
                'node_count': G.number_of_nodes(),
                'edge_count': G.number_of_edges(),
                'struct_total': struct_crime.metadata['total_amount'],
                'struct_sources': len(struct_crime.nodes_involved) - 1,
                'layer_initial': layer_crime.metadata['initial_amount'],
                'layer_final': layer_crime.metadata['final_amount'],
                'struct_evidence': len(struct_crime.metadata.get('evidence_artifacts', [])),
                'layer_evidence': len(layer_crime.metadata.get('evidence_artifacts', []))
            })
        
        # Compare results
        assert results[0] == results[1], "Full pipeline should be reproducible with same seed"
    
    def test_assessment_reproducibility(self, seed, difficulty):
        """Test that assessment scoring is reproducible."""
        from src.core.a2a_interface import (
            _calculate_pattern_score,
            _calculate_evidence_quality,
            _calculate_narrative_clarity,
            _calculate_completeness
        )
        
        # Same investigation data
        investigation_data = {
            'identified_crimes': [{'crime_type': 'structuring', 'nodes': [0]}],
            'suspicious_accounts': ['0', '1', '2'],
            'narrative': 'Investigation found structuring pattern with multiple deposits below threshold.',
            'transaction_ids': ['txn_1', 'txn_2'],
            'temporal_patterns': True,
            'amount_patterns': True
        }
        
        ground_truth = {
            'crimes': [
                {'crime_type': 'structuring', 'nodes_involved': [0, 1, 2], 'metadata': {'mule_id': 0}}
            ]
        }
        
        # Calculate scores twice
        scores1 = {
            'pattern': _calculate_pattern_score(
                investigation_data['identified_crimes'],
                ground_truth['crimes']
            ),
            'evidence': _calculate_evidence_quality(investigation_data),
            'narrative': _calculate_narrative_clarity(investigation_data),
            'completeness': _calculate_completeness(investigation_data, ground_truth)
        }
        
        scores2 = {
            'pattern': _calculate_pattern_score(
                investigation_data['identified_crimes'],
                ground_truth['crimes']
            ),
            'evidence': _calculate_evidence_quality(investigation_data),
            'narrative': _calculate_narrative_clarity(investigation_data),
            'completeness': _calculate_completeness(investigation_data, ground_truth)
        }
        
        # Scores should be identical
        assert scores1 == scores2, "Assessment scoring should be deterministic"
