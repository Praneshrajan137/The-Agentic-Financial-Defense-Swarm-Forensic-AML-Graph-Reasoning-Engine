# Product Requirements Document
## Green Financial Crime Agent - The Panopticon Protocol

**Version**: 1.0  
**Status**: Draft  
**Last Updated**: 2026-01-29

---

## 1. Executive Summary

### 1.1 Project Overview

**Project Name**: Green Financial Crime Agent  
**Codename**: The Panopticon Protocol  
**Mission**: Zero-Failure Synthetic Financial Crime Simulator

The Green Financial Crime Agent is the world's first true financial crime simulator, designed to generate mathematically consistent synthetic economies with surgically injected money laundering typologies. This system produces labeled training data for the Purple Agent (investigation/detection AI) without exposing real financial data or compromising privacy.

### 1.2 Objectives

1. Generate realistic scale-free financial transaction networks (1,000 nodes, ~10,000 edges)
2. Inject precise money laundering patterns (Structuring, Layering) with exact specifications
3. Produce labeled datasets for ML training and AML system testing
4. Expose data via Agent2Agent (A2A) protocol for seamless integration

### 1.3 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Graph Generation Time | <10 seconds | Automated benchmark |
| Memory Usage | <2GB | Memory profiler |
| Test Coverage | 90%+ | pytest-cov |
| Crime Label Accuracy | 100% | Validation scripts |
| Zero Failures | 50+ iterations | Execution log |

---

## 2. Problem Statement

### 2.1 Current Challenges

**AML Analyst Bottleneck**: Financial crime investigators spend 70-80% of their time on data gathering and preparation, leaving only 20-30% for actual analysis and investigation.

**Training Data Scarcity**: 
- Real financial crime data is sensitive and protected
- Labeled datasets are extremely rare
- Synthetic data lacks realistic patterns and correlations

**ML Model Limitations**:
- Models trained on unrealistic data perform poorly in production
- Lack of ground truth labels prevents proper evaluation
- Imbalanced datasets (crime is rare) cause detection issues

### 2.2 Impact

Without high-quality synthetic training data:
- AML systems generate excessive false positives (90%+ false positive rates)
- Analysts waste time investigating legitimate transactions
- Real crimes slip through due to alert fatigue
- Regulatory compliance suffers

---

## 3. Solution Overview

### 3.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GREEN FINANCIAL CRIME AGENT                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│   │   GENERATE  │────▶│   INJECT    │────▶│   EXPOSE    │          │
│   │   Graph     │     │   Crime     │     │   via A2A   │          │
│   └─────────────┘     └─────────────┘     └─────────────┘          │
│         │                   │                   │                   │
│   NetworkX +          Structuring +       HTTP/JSON-RPC +          │
│   Faker + SDV         Layering            Protobuf                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Core Components

1. **Graph Generator**: Creates scale-free financial networks using NetworkX
2. **Entity Generator**: Populates nodes with realistic data using Faker
3. **Correlation Engine**: Maintains statistical relationships using SDV
4. **Crime Injector**: Surgically injects Structuring and Layering patterns
5. **A2A Interface**: Exposes data via Agent2Agent protocol

### 3.3 Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Graph Operations | NetworkX 3.x | Scale-free graph generation |
| Entity Generation | Faker 18.x | Realistic names, companies, accounts |
| Correlations | SDV 1.x | Statistical relationship modeling |
| API Framework | FastAPI 0.100+ | HTTP/JSON-RPC interface |
| Serialization | Protobuf 4.x | 80% size reduction, 33x faster |
| Testing | pytest 7.x | TDD with 90%+ coverage |

---

## 4. User Stories

### US-001: Generate Scale-Free Graph

**As** the system  
**I need to** generate a scale-free graph with 1,000 nodes and ~10,000 edges  
**So that** I have a realistic baseline financial network

#### Acceptance Criteria

- [ ] Graph has exactly 1,000 nodes representing financial entities
- [ ] Graph has approximately 10,000 edges (±1,000 tolerance)
- [ ] Degree distribution follows power-law (scale-free property verified)
- [ ] Graph generation completes in less than 10 seconds
- [ ] Peak memory usage remains below 2GB
- [ ] Same seed produces identical graph (reproducibility)

#### Technical Specification

```python
# NetworkX Scale-Free Graph Configuration
nx.scale_free_graph(
    n=1000,
    alpha=0.41,  # P(new node connected to existing node)
    beta=0.54,   # P(edge between two existing nodes)
    gamma=0.05,  # P(new node connected from existing node)
    seed=42      # Reproducibility
)

# Parameter Constraint: alpha + beta + gamma = 1.0
# 0.41 + 0.54 + 0.05 = 1.00 ✓
```

#### Test Scenarios

```python
def test_graph_has_1000_nodes():
    G = generate_scale_free_graph(n_nodes=1000)
    assert G.number_of_nodes() == 1000

def test_graph_has_approximately_10000_edges():
    G = generate_scale_free_graph(n_nodes=1000)
    assert 9000 <= G.number_of_edges() <= 11000

def test_graph_is_scale_free():
    G = generate_scale_free_graph(n_nodes=1000)
    is_valid, metrics = validate_scale_free_distribution(G)
    assert is_valid
    assert metrics["hub_count"] > 0

def test_graph_generation_performance():
    import time
    start = time.time()
    G = generate_scale_free_graph(n_nodes=1000)
    elapsed = time.time() - start
    assert elapsed < 10.0
```

---

### US-002: Inject Structuring Crime

**As** the system  
**I need to** inject a structuring (smurfing) crime pattern  
**So that** the graph contains labeled money laundering activity

#### Acceptance Criteria

- [ ] Creates exactly 20 source nodes (smurfs)
- [ ] All 20 transfers target a single mule node
- [ ] All transaction amounts between $9,000 and $9,800
- [ ] All transfers occur within a 48-hour window
- [ ] All crime edges are labeled as "structuring"
- [ ] No amount equals or exceeds $10,000 (CTR threshold)

#### Technical Specification

```python
# Structuring (Smurfing) Configuration
STRUCTURING_CONFIG = {
    "num_sources": 20,           # Number of smurf accounts
    "min_amount": 9000.0,        # Minimum transfer amount
    "max_amount": 9800.0,        # Maximum transfer amount
    "time_window_hours": 48,     # All transfers within this window
    "target_nodes": 1            # Single mule account
}

# Pattern: Fan-In
#
#   Smurf_1 ($9,234) ─────┐
#   Smurf_2 ($9,567) ─────┤
#   Smurf_3 ($9,012) ─────┼────▶ MULE
#   ...                   │
#   Smurf_20 ($9,789) ────┘
#
# Total: ~$190,000 in 20 transactions, all below CTR threshold
```

#### Test Scenarios

```python
def test_structuring_creates_20_edges():
    G = generate_scale_free_graph()
    G, crime = inject_structuring(G)
    assert len(crime.edges_involved) == 20

def test_structuring_single_target():
    G = generate_scale_free_graph()
    G, crime = inject_structuring(G)
    targets = set(e[1] for e in crime.edges_involved)
    assert len(targets) == 1

def test_structuring_amounts_below_ctr():
    G = generate_scale_free_graph()
    G, crime = inject_structuring(G)
    for edge in crime.edges_involved:
        amount = G.edges[edge]["amount"]
        assert 9000 <= amount <= 9800
        assert amount < 10000  # CTR threshold

def test_structuring_within_time_window():
    G = generate_scale_free_graph()
    G, crime = inject_structuring(G)
    timestamps = [G.edges[e]["timestamp"] for e in crime.edges_involved]
    time_span = max(timestamps) - min(timestamps)
    assert time_span.total_seconds() <= 48 * 3600
```

---

### US-003: Inject Layering Crime

**As** the system  
**I need to** inject a layering crime pattern with decay  
**So that** the graph contains complex money trail obfuscation

#### Acceptance Criteria

- [ ] Creates a directed chain of transfers (no branches)
- [ ] Chain contains NO cycles (validated)
- [ ] Each hop has 2-5% decay in amount
- [ ] Amounts monotonically decrease along chain
- [ ] All crime edges are labeled as "layering"
- [ ] Chain length is configurable (default: 5 hops)

#### Technical Specification

```python
# Layering Configuration
LAYERING_CONFIG = {
    "chain_length": 5,           # Number of intermediary hops
    "min_decay": 0.02,           # Minimum decay rate (2%)
    "max_decay": 0.05,           # Maximum decay rate (5%)
    "initial_amount": 100000.0   # Starting amount
}

# Pattern: Chain with Decay
#
#   Source ──($100,000)──▶ Layer_1 ──($97,500)──▶ Layer_2 ──($95,063)──▶ ...
#                           -2.5%                   -2.5%
#
# Decay Formula: amount_n = amount_(n-1) * (1 - decay_rate)
# Where: decay_rate ∈ [0.02, 0.05]
```

#### Test Scenarios

```python
def test_layering_no_cycles():
    G = generate_scale_free_graph()
    G, crime = inject_layering(G)
    is_valid = validate_no_cycles(G, crime.edges_involved)
    assert is_valid

def test_layering_forms_chain():
    G = generate_scale_free_graph()
    G, crime = inject_layering(G, LayeringConfig(chain_length=5))
    assert len(crime.edges_involved) == 5

def test_layering_decay_rate():
    G = generate_scale_free_graph()
    G, crime = inject_layering(G)
    amounts = [G.edges[e]["amount"] for e in crime.edges_involved]
    for i in range(1, len(amounts)):
        decay = 1 - (amounts[i] / amounts[i-1])
        assert 0.02 <= decay <= 0.05

def test_layering_monotonic_decrease():
    G = generate_scale_free_graph()
    G, crime = inject_layering(G)
    amounts = [G.edges[e]["amount"] for e in crime.edges_involved]
    assert amounts == sorted(amounts, reverse=True)
```

---

### US-004: Expose via A2A Protocol

**As** the system  
**I need to** expose data via Agent2Agent protocol  
**So that** the Purple Agent can consume the synthetic data

#### Acceptance Criteria

- [ ] `agent.json` manifest file is created and accessible
- [ ] Health endpoint (`/health`) returns status
- [ ] Graph generation endpoint works via API
- [ ] Crime injection endpoint works via API
- [ ] Data export available in JSON format
- [ ] Data export available in Protobuf format (80% smaller)

#### Technical Specification

```python
# Agent Manifest (agent.json)
{
    "name": "green-financial-crime-agent",
    "version": "0.1.0",
    "description": "Synthetic financial crime data generator",
    "capabilities": [
        "generate_graph",
        "inject_structuring",
        "inject_layering",
        "export_data"
    ],
    "endpoints": {
        "health": "/health",
        "manifest": "/agent.json",
        "graph": "/api/v1/graph",
        "crimes": "/api/v1/crimes",
        "export": "/api/v1/graph/export"
    }
}

# API Endpoints
GET  /health              → {"status": "healthy", "version": "0.1.0"}
GET  /agent.json          → Agent manifest
POST /api/v1/graph        → Generate new graph
POST /api/v1/crimes       → Inject crime pattern
GET  /api/v1/graph/export → Export graph (JSON or Protobuf)
GET  /api/v1/crimes/labels → Get crime labels for training
```

#### Test Scenarios

```python
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_agent_manifest():
    response = client.get("/agent.json")
    assert response.status_code == 200
    assert "capabilities" in response.json()

def test_graph_generation_endpoint():
    response = client.post("/api/v1/graph", json={"n_nodes": 100})
    assert response.status_code == 200
    assert response.json()["nodes"] == 100

def test_export_protobuf_smaller():
    json_response = client.get("/api/v1/graph/export?format=json")
    proto_response = client.get("/api/v1/graph/export?format=protobuf")
    assert len(proto_response.content) < len(json_response.content) * 0.3
```

---

## 5. Technical Specifications

### 5.1 Data Models

#### Node (Entity) Schema

```python
@dataclass
class FinancialEntity:
    id: int                      # Unique identifier
    entity_type: str             # "person" | "company" | "bank"
    name: str                    # Generated name
    account_number: str          # Masked account number
    country: str                 # ISO 3166-1 alpha-2
    risk_score: float            # 0.0 - 1.0
    created_at: datetime         # Entity creation timestamp
```

#### Edge (Transaction) Schema

```python
@dataclass
class Transaction:
    transaction_id: str          # UUID
    source: int                  # Source node ID
    target: int                  # Target node ID
    amount: float                # Transaction amount
    currency: str                # ISO 4217 (default: USD)
    timestamp: datetime          # Transaction timestamp
    transaction_type: str        # "wire" | "ach" | "cash"
    label: str                   # "legitimate" | "structuring" | "layering"
```

### 5.2 Graph Generation Algorithm

```python
def generate_financial_network(config: GraphConfig) -> nx.DiGraph:
    """
    Generate a scale-free financial transaction network.
    
    Algorithm:
    1. Create scale-free graph structure (NetworkX)
    2. Assign entity attributes to nodes (Faker)
    3. Assign transaction attributes to edges (SDV correlations)
    4. Validate graph properties
    """
    # Step 1: Structure
    G = nx.scale_free_graph(
        n=config.n_nodes,
        alpha=config.alpha,
        beta=config.beta,
        gamma=config.gamma,
        seed=config.seed
    )
    
    # Step 2: Node attributes
    for node in G.nodes():
        G.nodes[node].update(generate_entity(faker_instance))
    
    # Step 3: Edge attributes
    for u, v in G.edges():
        G.edges[u, v].update(generate_transaction(sdv_synthesizer))
    
    # Step 4: Validation
    assert validate_graph_structure(G)
    assert validate_scale_free_distribution(G)
    
    return G
```

### 5.3 Crime Injection Algorithms

#### Structuring Injection

```python
def inject_structuring(G: nx.DiGraph, config: StructuringConfig) -> InjectedCrime:
    """
    Inject structuring pattern: fan-in to single mule.
    
    Algorithm:
    1. Select or create mule node (high connectivity preferred)
    2. Create 20 source nodes
    3. For each source:
       a. Generate amount in [$9000, $9800]
       b. Generate timestamp within 48hr window
       c. Create edge: source -> mule
       d. Label edge as "structuring"
    4. Validate all amounts < $10,000
    """
    mule = select_or_create_mule(G)
    sources = create_smurf_nodes(G, config.num_sources)
    
    edges = []
    base_time = datetime.now()
    
    for source in sources:
        amount = random.uniform(config.min_amount, config.max_amount)
        timestamp = base_time + timedelta(
            seconds=random.randint(0, config.time_window_hours * 3600)
        )
        
        G.add_edge(source, mule, 
                   amount=amount,
                   timestamp=timestamp,
                   label="structuring")
        edges.append((source, mule))
    
    return InjectedCrime(
        crime_type="structuring",
        nodes_involved=[mule] + sources,
        edges_involved=edges,
        metadata={"total_amount": sum(G.edges[e]["amount"] for e in edges)}
    )
```

#### Layering Injection

```python
def inject_layering(G: nx.DiGraph, config: LayeringConfig) -> InjectedCrime:
    """
    Inject layering pattern: chain with decay.
    
    Algorithm:
    1. Create chain_length + 1 nodes
    2. For each hop (0 to chain_length - 1):
       a. Calculate amount with decay
       b. Create edge: node[i] -> node[i+1]
       c. Label edge as "layering"
    3. Validate no cycles exist
    """
    nodes = create_chain_nodes(G, config.chain_length + 1)
    
    edges = []
    amount = config.initial_amount
    
    for i in range(config.chain_length):
        decay = random.uniform(config.min_decay, config.max_decay)
        amount = amount * (1 - decay)
        
        G.add_edge(nodes[i], nodes[i + 1],
                   amount=amount,
                   label="layering")
        edges.append((nodes[i], nodes[i + 1]))
    
    # Validate no cycles
    assert validate_no_cycles(G, edges)
    
    return InjectedCrime(
        crime_type="layering",
        nodes_involved=nodes,
        edges_involved=edges,
        metadata={"decay_total": 1 - (amount / config.initial_amount)}
    )
```

---

## 6. Non-Functional Requirements

### 6.1 Performance

| Requirement | Target | Measurement Method |
|-------------|--------|-------------------|
| Graph generation (1k nodes) | <10 seconds | `time.time()` benchmark |
| Crime injection | <1 second | `time.time()` benchmark |
| API response time | <500ms | FastAPI middleware |
| Protobuf serialization | 33x faster than JSON | Benchmark comparison |

### 6.2 Scalability

| Scale | Nodes | Edges | Expected Time | Memory |
|-------|-------|-------|---------------|--------|
| Small | 1,000 | 10,000 | <10s | <500MB |
| Medium | 10,000 | 100,000 | <60s | <2GB |
| Large | 100,000 | 1,000,000 | <10min | <8GB |

### 6.3 Reliability

- **Zero failures** across 50+ consecutive iterations
- **100% reproducibility** with same seed
- **Graceful degradation** on resource constraints
- **Comprehensive error handling** with meaningful messages

### 6.4 Maintainability

- **Test coverage**: 90%+ (enforced in CI)
- **Type hints**: All functions fully typed
- **Documentation**: Docstrings for all public APIs
- **Code style**: Black + Ruff enforced

---

## 7. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Context rot during long sessions | High | High | Persistent state (prd.json, progress.txt) |
| Data quality issues | Medium | High | Statistical validation scripts |
| Performance degradation at scale | Medium | Medium | Protobuf serialization, lazy loading |
| Unrealistic graph topology | Low | High | Validated scale-free distribution |
| Crime pattern detection leakage | Low | Medium | Isolated crime injection, no real data |

---

## 8. Implementation Roadmap

### Phase 1: Foundation (Current)
- [x] Project structure
- [x] Cursor rules configuration
- [x] PRD documentation
- [x] Architecture documentation

### Phase 2: Graph Generation
- [ ] Write tests for graph generator
- [ ] Implement `generate_scale_free_graph()`
- [ ] Implement Faker entity generation
- [ ] Implement SDV correlation modeling
- [ ] Validate scale-free properties

### Phase 3: Crime Injection
- [ ] Write tests for crime injector
- [ ] Implement `inject_structuring()`
- [ ] Implement `inject_layering()`
- [ ] Implement cycle detection
- [ ] Implement crime labeling

### Phase 4: A2A Interface
- [ ] Write tests for API endpoints
- [ ] Implement FastAPI application
- [ ] Implement JSON export
- [ ] Implement Protobuf export
- [ ] Create agent.json manifest

### Phase 5: Integration & Validation
- [ ] End-to-end integration tests
- [ ] Performance benchmarking
- [ ] Memory profiling
- [ ] Documentation review
- [ ] Release preparation

---

## 9. Glossary

| Term | Definition |
|------|------------|
| **A2A Protocol** | Agent2Agent - communication protocol for AI agents |
| **CTR** | Currency Transaction Report - required for transactions ≥$10,000 |
| **Layering** | Money laundering technique using chain transfers with decay |
| **Mule** | Account used to receive illicit funds |
| **Scale-Free Graph** | Network where degree distribution follows power law |
| **SDV** | Synthetic Data Vault - library for generating correlated data |
| **Smurfing** | Synonym for structuring |
| **Structuring** | Breaking large transactions into smaller ones to avoid CTR |

---

## 10. Appendix

### A. References

1. Barabási-Albert model for scale-free networks
2. FinCEN Bank Secrecy Act requirements
3. Google A2A Protocol specification
4. SDV Gaussian Copula documentation

### B. Change Log

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-29 | System | Initial PRD creation |
