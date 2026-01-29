# Green Financial Crime Agent - Architecture

## Overview

The Panopticon Protocol is a Zero-Failure Synthetic Financial Crime Simulator designed to generate mathematically consistent economies with surgically injected money laundering typologies.

## System Architecture

### Client-Server Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐                          ┌─────────────────┐          │
│  │  Purple Agent   │◄────── A2A Protocol ────►│   Green Agent   │          │
│  │    (Client)     │        HTTP/JSON-RPC     │    (Server)     │          │
│  └─────────────────┘                          └─────────────────┘          │
│         │                                            │                      │
│         │ Requests:                                  │ Provides:            │
│         │ • get_transactions                         │ • Synthetic graphs   │
│         │ • get_kyc_profile                          │ • Crime patterns     │
│         │ • investigation_assessment                 │ • Ground truth       │
│         │                                            │                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        GREEN FINANCIAL CRIME AGENT                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   Graph      │───▶│    Crime     │───▶│     A2A      │              │
│  │  Generator   │    │   Injector   │    │  Interface   │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  NetworkX    │    │ Structuring  │    │  HTTP/JSON   │              │
│  │  Faker/SDV   │    │  Layering    │    │  Protobuf    │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Graph Generator (`src/core/graph_generator.py`)

**Purpose**: Generate scale-free financial transaction networks.

**Technology**: NetworkX

**Input**: Configuration (nodes, edges, parameters)

**Process**: NetworkX scale_free_graph + Faker + SDV

**Output**: Baseline economy graph (pickle format)

**Key Parameters**:
- `n_nodes`: 1,000 (entities)
- `alpha`: 0.41 (new node → existing node probability)
- `beta`: 0.54 (edge between existing nodes probability)
- `gamma`: 0.05 (existing node → new node probability)

**Locale Support**: Faker with locales `en_US`, `en_GB`, `en_IN` for diverse entity generation.

### 2. Crime Injector (`src/core/crime_injector.py`)

**Purpose**: Surgically inject money laundering patterns into legitimate graphs.

**Input**: Baseline graph, crime type (Structuring/Layering)

**Process**: Surgical subgraph insertion with ground truth ledger

**Output**: Poisoned graph + metadata (JSON)

**Supported Patterns**:

| Pattern | Description | Key Metrics |
|---------|-------------|-------------|
| Structuring | Fan-in smurfing | 20 sources, 1 mule, <$10k each |
| Layering | Chain with decay | 2-5% decay per hop, no cycles |

### 3. A2A Interface (`src/core/a2a_interface.py`)

**Purpose**: Expose data via Agent2Agent protocol for Purple Agent integration.

**Input**: HTTP/JSON-RPC requests

**Process**: Tool dispatch (get_transactions, get_kyc_profile)

**Output**: Incremental data (JSON response)

**Endpoints**:
- `POST /a2a/investigation_assessment` - Investigation simulation
- `POST /a2a/tools/get_transactions` - Transaction history
- `POST /a2a/tools/get_kyc_profile` - KYC profile lookup
- `GET /health` - Health check
- `GET /agent.json` - Manifest for discovery

**Serialization**:
- JSON (human-readable)
- Protobuf (80% size reduction, 33x faster)

## Data Flow

```
┌──────────┐     ┌─────────────────┐     ┌────────────────┐
│  Config  │────▶│ Graph Generator │────▶│ Baseline Graph │
└──────────┘     └─────────────────┘     └───────┬────────┘
                                                 │
                                                 ▼
                                    ┌────────────────────┐
                                    │   Crime Injector   │
                                    └─────────┬──────────┘
                                              │
                         ┌────────────────────┼────────────────────┐
                         ▼                    ▼                    ▼
                 ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
                 │Poisoned Graph│    │ Ground Truth  │    │   Metadata   │
                 │   (pickle)   │    │    (JSON)     │    │    (JSON)    │
                 └──────┬───────┘    └───────────────┘    └──────────────┘
                        │
                        ▼
               ┌─────────────────┐
               │  A2A Interface  │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ Client (Purple  │
               │     Agent)      │
               └─────────────────┘
```

## Ralph Wiggum Execution Pattern

The Ralph Wiggum pattern prevents context rot through iterative execution cycles. While traditionally implemented as a bash wrapper, this project uses Cursor's native agent loop for the same effect.

### Execution Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                    RALPH WIGGUM ITERATION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. READ STATE                                                  │
│     ├── prd.json (user stories, tasks)                         │
│     ├── progress.txt (execution log)                           │
│     └── ARCHITECTURE.md (design reference)                      │
│                                                                 │
│  2. EXECUTE ONE ATOMIC TASK                                     │
│     ├── Find highest priority incomplete story                  │
│     ├── Write failing test (TDD)                               │
│     ├── Implement minimal code to pass                         │
│     └── Run test suite                                         │
│                                                                 │
│  3. UPDATE STATE                                                │
│     ├── Mark task complete in prd.json                         │
│     ├── Append to progress.txt                                 │
│     └── Commit changes                                         │
│                                                                 │
│  4. CHECK COMPLETION                                            │
│     └── If all stories pass: signal completion                 │
│                                                                 │
│  5. RESTART FRESH (context reset)                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Benefits

- **Zero Context Rot**: Fresh context each iteration
- **Persistent Progress**: State survives restarts
- **Atomic Tasks**: One focused task per cycle
- **Auditable**: Full history in progress.txt

## Data Models

### Node Schema (Entity)

```python
{
    "id": str,                    # Unique identifier
    "entity_type": str,           # "person" | "company" | "bank"
    "name": str,                  # Generated name
    "company": str,               # Company name (if applicable)
    "address": str,               # Full address
    "account_number": str,        # Masked account number
    "swift": str,                 # SWIFT/BIC code
    "iban": str,                  # IBAN (if applicable)
    "country": str,               # ISO 3166-1 alpha-2
    "risk_score": float,          # 0.0 - 1.0
    "created_at": datetime        # Entity creation timestamp
}
```

### Edge Schema (Transaction)

```python
{
    "transaction_id": str,        # UUID
    "source": str,                # Source node ID
    "target": str,                # Target node ID
    "amount": float,              # Transaction amount
    "currency": str,              # ISO 4217 (default: USD)
    "timestamp": datetime,        # Transaction timestamp
    "transaction_type": str,      # "wire" | "ach" | "cash"
    "label": str                  # "legitimate" | "structuring" | "layering"
}
```

### Ground Truth Ledger

```python
{
    "crime_type": str,            # "structuring" | "layering"
    "mule_id": str,               # Target node (structuring)
    "chain": list,                # Node chain (layering)
    "sources": list,              # Source nodes
    "total_amount": float,        # Sum of crime amounts
    "timestamp_range": {
        "start": datetime,
        "end": datetime
    }
}
```

## Technology Decisions

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Graph Operations | NetworkX 3.x | Scale-free topology, power-law distribution |
| Entity Generation | Faker 18.x | Localized generation (en_GB, en_IN) |
| Correlations | SDV 1.x | Multivariate correlation (Gaussian Copulas) |
| API Framework | FastAPI 0.100+ | Async, automatic OpenAPI docs |
| Serialization | Protobuf 4.x | 80% size reduction, 33x faster |
| Testing | pytest 7.x | TDD with 90%+ coverage |

## Directory Structure

```
green-agent/
├── .cursor/rules/           # Cursor agent guidance rules
├── .claude/skills/          # Claude skill definitions
│   ├── financial-crime/
│   │   ├── SKILL.md
│   │   └── scripts/
│   └── data-generation/
│       ├── SKILL.md
│       └── scripts/
├── src/
│   ├── core/               # Core modules
│   │   ├── graph_generator.py
│   │   ├── crime_injector.py
│   │   └── a2a_interface.py
│   └── utils/              # Utilities
│       └── validators.py
├── scripts/                # Deterministic operations
├── tests/                  # TDD test suite
├── outputs/                # Generated graphs and data
├── prd.json               # Machine state
├── progress.txt           # Execution log
├── prompt.md              # Ralph agent instructions
├── API-spec.yml           # OpenAPI specification
└── requirements.txt       # Dependencies
```

## Design Principles

### 1. Algorithm Over AI

All mathematical operations use deterministic Python scripts, not AI/LLM:
- Graph generation: `scripts/generate_graph.py`
- Crime injection: `scripts/inject_structuring.py`, `scripts/inject_layering.py`
- Validation: `scripts/detect_cycles.py`

### 2. Test-Driven Development (TDD)

1. Write failing tests first
2. Implement minimal code to pass
3. Refactor while keeping tests green
4. Target: 90%+ coverage

### 3. Persistent State

Machine-readable state for iteration tracking:
- `prd.json`: User stories, tasks, configuration
- `progress.txt`: Execution log with timestamps

### 4. Zero Context Rot

Each iteration:
1. Reads state from disk
2. Executes ONE atomic task
3. Updates state
4. Proceeds to next task

## Security Considerations

### Data Security

1. **Synthetic Data Only**: No real PII or financial data
2. **Labeled Output**: All crime patterns are explicitly labeled
3. **Deterministic**: Same seed produces identical output
4. **Sandboxed**: No external network calls during generation

### API Security

1. **Rate Limiting**: Stripe's 4-tier model
   - Request Rate Limiter (Token Bucket)
   - Concurrent Request Limiter
   - Fleet Usage Load Shedder
   - Worker Utilization Load Shedder

2. **Input Validation**: All parameters validated via Pydantic
   - Type checking
   - Range validation
   - Schema enforcement

3. **No PII Exposure**: Purely random synthetic data

### Rate Limiting Implementation

```python
# Token Bucket Configuration
RATE_LIMIT_CONFIG = {
    "requests_per_minute": 60,
    "burst_size": 10,
    "concurrent_limit": 5
}
```

## Performance Requirements

| Metric | Target | Hard Limit |
|--------|--------|------------|
| Graph Generation (1k nodes) | <5s | 10s |
| Crime Injection | <1s | 5s |
| Memory Usage | <1GB | 2GB |
| API Response | <100ms | 500ms |
| Test Coverage | >90% | 85% min |

## Future Extensions

1. **Additional Crime Types**: Round-tripping, trade-based ML
2. **Multi-Graph Support**: Multiple interconnected networks
3. **Real-time Streaming**: WebSocket interface
4. **Graph Database Export**: Neo4j, TigerGraph formats
5. **Multi-locale Expansion**: Additional Faker locales
