---
name: financial-crime
description: Expert knowledge on AML typologies, crime injection, and graph-based detection
version: 1.0.0
---

# Financial Crime Detection Skill

This skill provides expert knowledge on anti-money laundering (AML) typologies and synthetic crime generation for the Green Financial Crime Agent.

## Core Concepts

### Structuring (Smurfing)

**Definition**: Breaking large deposits into smaller amounts to evade the $10,000 Currency Transaction Report (CTR) threshold.

**Topology**: Fan-in pattern (multiple sources → one mule)

```
    Smurf_1 ($9,234) ─────┐
    Smurf_2 ($9,567) ─────┤
    Smurf_3 ($9,012) ─────┼────▶ MULE
    ...                   │
    Smurf_20 ($9,789) ────┘
```

**Detection Heuristics**:
- Amount clustering near threshold ($9,000-$9,800)
- Temporal compression (24-48 hour window)
- Unusual geographic diversity of sources
- Multiple new accounts sending to single recipient

**Configuration**:
```python
STRUCTURING_CONFIG = {
    "num_sources": 20,           # Number of smurf accounts
    "min_amount": 9000.0,        # Minimum transfer amount
    "max_amount": 9800.0,        # Maximum transfer amount (below CTR)
    "time_window_hours": 48,     # All transfers within this window
    "target_nodes": 1            # Single mule account
}
```

### Layering

**Definition**: Obscuring the audit trail through complex chains of transfers with value decay.

**Topology**: Directed path with value decay

```
Source ──($100,000)──▶ Layer_1 ──($97,500)──▶ Layer_2 ──($95,063)──▶ ... ──▶ Dest
              -2.5%                  -2.5%                  -2.5%
```

**Detection Heuristics**:
- Rapid movement (minimal time between hops)
- Consistent decay pattern (2-5% per hop)
- Chain length 5-10 hops typical
- No cycles in transaction path

**Configuration**:
```python
LAYERING_CONFIG = {
    "chain_length_min": 5,       # Minimum number of hops
    "chain_length_max": 7,       # Maximum number of hops
    "min_decay": 0.02,           # Minimum decay rate (2%)
    "max_decay": 0.05,           # Maximum decay rate (5%)
    "initial_amount": 100000.0,  # Starting amount
    "time_window_hours": 24      # All hops within this window
}
```

## Tools

### generate_graph.py

Generate scale-free financial networks using NetworkX.

**Location**: `.claude/skills/financial-crime/scripts/generate_graph.py`

**Usage**:
```bash
python scripts/generate_graph.py --nodes 1000 --output outputs/baseline.pkl
```

**Key Function**:
```python
def generate_scale_free_graph(
    n: int,
    alpha: float = 0.41,
    beta: float = 0.54,
    gamma: float = 0.05,
    seed: Optional[int] = None
) -> nx.DiGraph:
    """Generate scale-free graph with realistic attributes."""
```

### inject_structuring.py

Inject smurfing pattern into graph.

**Location**: `.claude/skills/financial-crime/scripts/inject_structuring.py`

**Usage**:
```bash
python scripts/inject_structuring.py --graph outputs/baseline.pkl --mule node_42 --output outputs/poisoned.pkl
```

**Key Function**:
```python
def inject_structuring(
    G: nx.DiGraph,
    mule_id: str,
    num_sources: int = 20,
    seed: Optional[int] = None
) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """Inject structuring crime pattern."""
```

### inject_layering.py

Inject layering chain pattern.

**Location**: `.claude/skills/financial-crime/scripts/inject_layering.py`

**Usage**:
```bash
python scripts/inject_layering.py --graph outputs/baseline.pkl --source node_10 --dest node_99 --output outputs/poisoned.pkl
```

**Key Function**:
```python
def inject_layering(
    G: nx.DiGraph,
    source: str,
    dest: str,
    chain_length: int = 6,
    seed: Optional[int] = None
) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """Inject layering crime pattern."""
```

### detect_cycles.py

Validate graph for cycles (especially in crime chains).

**Location**: `.claude/skills/financial-crime/scripts/detect_cycles.py`

**Usage**:
```bash
python scripts/detect_cycles.py --graph outputs/poisoned.pkl
```

**Key Function**:
```python
def detect_cycles(G: nx.DiGraph) -> List[List[str]]:
    """Detect all cycles in graph."""
```

## Ground Truth Ledger Format

All crime injections produce a ground truth JSON:

```json
{
    "crime_type": "structuring",
    "mule_id": "node_42",
    "sources": [
        {
            "id": "smurf_0",
            "amount": 9234.56,
            "timestamp": "2026-01-15T10:30:00Z"
        }
    ],
    "total_amount": 189456.78,
    "timestamp_range": {
        "start": "2026-01-15T08:00:00Z",
        "end": "2026-01-16T14:00:00Z"
    }
}
```

## Best Practices

1. **Always generate ground truth ledgers** - Required for Purple Agent training
2. **Validate graph topology after injection** - Check for unintended cycles
3. **Use consistent seed for reproducibility** - Same seed = identical output
4. **Store metadata alongside graphs** - Keep JSON ground truth with pickle
5. **Label all crime edges** - Use `label` attribute on edges

## Anti-Patterns to Avoid

1. **DO NOT** use random amounts outside specified ranges
2. **DO NOT** create cycles in layering chains
3. **DO NOT** exceed 48-hour window for structuring
4. **DO NOT** use AI for amount/timestamp generation (use deterministic scripts)
5. **DO NOT** modify baseline graph structure outside injection functions
