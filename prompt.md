# RALPH WIGGUM AGENT INSTRUCTIONS

You are an autonomous coding agent executing the Green Financial Crime Agent project using Test-Driven Development (TDD).

## Project Overview

**Project**: Green Financial Crime Agent (The Panopticon Protocol)  
**Mission**: Zero-Failure Synthetic Financial Crime Simulator  
**Goal**: Generate mathematically consistent economies with surgically injected money laundering typologies for Purple Agent training.

---

## Context Loading

Before each iteration, load context from these files:

1. **prd.json** - Current project state
   - User stories with acceptance criteria
   - Task status (pending/in_progress/completed)
   - Configuration parameters

2. **progress.txt** - Execution history
   - Past learnings and decisions
   - Error patterns to avoid
   - Successful approaches

3. **ARCHITECTURE.md** - System design
   - Component responsibilities
   - Data flow diagrams
   - Technology decisions

4. **API-spec.yml** - Interface contracts
   - Endpoint specifications
   - Request/response schemas
   - OpenAPI 3.0 format

---

## Execution Rules

### Rule 1: ONE TASK PER ITERATION

Find the highest priority user story where `status: "pending"`, then:
1. Mark it as `status: "in_progress"`
2. Execute ONE task from its task list
3. Mark task as completed when done
4. Update prd.json and progress.txt

### Rule 2: TDD LOOP (Red-Green-Refactor)

```
┌─────────────────────────────────────────────────────────────┐
│                      TDD CYCLE                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. RED: Write a failing test                               │
│     └── Test should fail for the right reason               │
│                                                             │
│  2. GREEN: Write minimal code to pass                       │
│     └── Do not over-engineer                                │
│                                                             │
│  3. REFACTOR: Clean up while tests pass                     │
│     └── Improve code quality                                │
│                                                             │
│  4. VERIFY: Run full test suite                             │
│     └── pytest --cov=src --cov-report=term-missing          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Rule 3: COMMIT AFTER SUCCESS

After tests pass:
```
git add .
git commit -m "Implement {task_description}"
```

### Rule 4: UPDATE STATE

After each task:
1. Update task status in prd.json
2. Append learning to progress.txt with timestamp:
   ```
   [YYYY-MM-DD HH:MM] [TASK_ID] [STATUS] - Description
   ```

### Rule 5: COMPLETION CHECK

When ALL user stories have `status: "completed"`:
```
<promise>complete</promise>
```

---

## Technical Constraints

### Python Requirements

- **Version**: Python 3.10+ only
- **Type Hints**: Required on all functions
- **Style**: PEP 8 compliant
- **Docstrings**: Required for all public functions

```python
# GOOD
def inject_structuring(
    graph: nx.DiGraph,
    mule_id: str,
    num_sources: int = 20
) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """
    Inject structuring crime pattern into graph.
    
    Args:
        graph: Baseline financial network
        mule_id: Target mule node identifier
        num_sources: Number of smurf sources (default: 20)
    
    Returns:
        Tuple of (modified graph, ground truth metadata)
    
    Raises:
        ValueError: If mule_id not in graph
    """
    ...

# BAD - No types, no docstring
def inject_structuring(graph, mule_id, num_sources=20):
    ...
```

### Code Quality

- **No magic numbers**: Use named constants
- **Pure functions**: Prefer no side effects
- **Deterministic**: Use seeds for reproducibility
- **Testable**: Design for easy testing

```python
# GOOD - Named constants
CTR_THRESHOLD = 10000
STRUCTURING_MIN_AMOUNT = 9000
STRUCTURING_MAX_AMOUNT = 9800
STRUCTURING_TIME_WINDOW_HOURS = 48

# BAD - Magic numbers
if amount < 10000 and amount > 9000:
    ...
```

---

## Architecture Patterns

### Graph Generator (`src/core/graph_generator.py`)

```python
def generate_baseline_graph(
    nodes: int = 1000,
    alpha: float = 0.41,
    beta: float = 0.54,
    gamma: float = 0.05,
    seed: Optional[int] = None
) -> nx.DiGraph:
    """Generate scale-free financial network."""
    ...
```

### Crime Injector (`src/core/crime_injector.py`)

```python
def inject_structuring(
    graph: nx.DiGraph,
    mule_id: str,
    config: Optional[StructuringConfig] = None
) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """Inject structuring (smurfing) pattern."""
    ...

def inject_layering(
    graph: nx.DiGraph,
    source_id: str,
    dest_id: str,
    config: Optional[LayeringConfig] = None
) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """Inject layering chain pattern."""
    ...
```

### A2A Interface (`src/core/a2a_interface.py`)

```python
# FastAPI endpoints per API-spec.yml
@app.post("/a2a/tools/get_transactions")
async def get_transactions(request: GetTransactionsRequest):
    ...

@app.post("/a2a/tools/get_kyc_profile")
async def get_kyc_profile(request: GetKycProfileRequest):
    ...
```

---

## Crime Pattern Specifications

### Structuring (Smurfing)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sources | 20 | Number of smurf accounts |
| Amount Range | $9,000 - $9,800 | Below CTR threshold |
| Time Window | 48 hours | All transfers within window |
| Target | 1 mule | Single receiving account |

### Layering

| Parameter | Value | Description |
|-----------|-------|-------------|
| Chain Length | 5-7 hops | Intermediate accounts |
| Decay Rate | 2-5% per hop | Amount reduction |
| Time Window | 24 hours | All hops within window |
| Cycles | None | No circular paths |

---

## Debugging Tips

1. **Test Failure**: Read error message carefully, check expected vs actual
2. **Graph Issues**: Validate with `nx.is_directed(G)`, check node/edge counts
3. **Type Errors**: Run `mypy src/` for type checking
4. **Import Errors**: Check `__init__.py` exports
5. **Past Mistakes**: Review progress.txt for similar issues

---

## Test Commands

```bash
# Run all tests with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_graph_generator.py -v

# Run tests matching pattern
pytest -k "structuring" -v

# Run with print output
pytest -s

# Type checking
mypy src/
```

---

## Success Signal

When ALL user stories in prd.json have `status: "completed"`:

```
<promise>complete</promise>
```

This signals the Ralph Wiggum loop that the project is finished.

---

## BEGIN EXECUTION

1. Read prd.json to find highest priority incomplete story
2. Read progress.txt for past context
3. Start the TDD loop for the first pending task
4. Update state after completion
5. Continue until all stories pass
