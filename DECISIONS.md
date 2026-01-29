# Architectural Decision Records (ADR)

This document records the key architectural decisions made for the Green Financial Crime Agent project.

---

## ADR-001: Use NetworkX for Graph Generation

**Status**: Accepted

**Context**:
We need to generate scale-free graphs representing financial transaction networks. Options considered:
- NetworkX (Python)
- igraph (Python/C)
- Neo4j (Graph Database)
- Custom implementation

**Decision**:
Use NetworkX with `scale_free_graph()` function.

**Rationale**:
1. Native Python integration
2. Built-in scale-free graph algorithms
3. Extensive documentation and community
4. Sufficient performance for 1k-10k scale
5. Easy serialization to JSON/Protobuf

**Consequences**:
- (+) Fast development
- (+) Extensive graph algorithms available
- (-) May need optimization for >100k nodes

---

## ADR-002: Faker + SDV for Synthetic Data

**Status**: Accepted

**Context**:
Need realistic entity data (names, accounts, companies) with statistical correlations.

**Decision**:
Use Faker for entity generation, SDV (Gaussian Copulas) for correlations.

**Rationale**:
1. Faker provides locale-aware realistic data
2. SDV maintains statistical properties
3. Both support reproducible seeding
4. Industry-standard tools

**Consequences**:
- (+) Realistic output
- (+) Reproducible with seeds
- (-) SDV has learning curve

---

## ADR-003: Algorithm Over AI Principle

**Status**: Accepted

**Context**:
Some operations could use AI/LLM, but need determinism for testing and reproducibility.

**Decision**:
All mathematical operations (graph generation, crime injection, validation) MUST use deterministic Python scripts.

**Rationale**:
1. Reproducibility with same seed
2. Testable with exact assertions
3. No API costs for generation
4. Faster execution

**Consequences**:
- (+) 100% reproducible output
- (+) Fast and free execution
- (-) More explicit coding required

---

## ADR-004: Protobuf for Serialization

**Status**: Accepted

**Context**:
Need to serialize graph data efficiently for A2A protocol.

**Decision**:
Support both JSON (human-readable) and Protobuf (performance).

**Rationale**:
1. Protobuf: 80% size reduction vs JSON
2. Protobuf: 33x faster deserialization
3. JSON: Easy debugging and manual inspection
4. Both formats for different use cases

**Consequences**:
- (+) Optimal for different scenarios
- (-) Maintain two serialization paths

---

## ADR-005: FastAPI for A2A Interface

**Status**: Accepted

**Context**:
Need HTTP/JSON-RPC interface for Agent2Agent protocol.

**Decision**:
Use FastAPI with Pydantic models.

**Rationale**:
1. Automatic OpenAPI documentation
2. Built-in validation with Pydantic
3. Async support for performance
4. Modern Python best practices

**Consequences**:
- (+) Self-documenting API
- (+) Type safety
- (-) Requires async understanding

---

## ADR-006: TDD Methodology

**Status**: Accepted

**Context**:
Need high confidence in crime injection accuracy.

**Decision**:
Strict Test-Driven Development with 90%+ coverage target.

**Rationale**:
1. Crime patterns must be exactly correct
2. Prevents regression
3. Documents expected behavior
4. Enables fearless refactoring

**Consequences**:
- (+) High reliability
- (+) Living documentation
- (-) Slower initial development

---

## ADR-007: Cursor Native Agent Loop (No Ralph Wiggum)

**Status**: Accepted

**Context**:
Original spec suggested bash wrapper for die-and-restart cycles. User is on Windows.

**Decision**:
Use Cursor's native agent loop instead of external shell wrapper.

**Rationale**:
1. Windows compatibility
2. Better IDE integration
3. Cursor handles context management
4. Simpler architecture

**Consequences**:
- (+) Platform independent
- (+) Integrated experience
- (-) Less control over iteration timing

---

## ADR-008: .cursor/rules/ for Agent Guidance

**Status**: Accepted

**Context**:
Original spec used `.claude/skills/`. Need Cursor-compatible approach.

**Decision**:
Use `.cursor/rules/` with `.mdc` files for agent guidance.

**Rationale**:
1. Native Cursor format
2. Glob-based file matching
3. Can be conditionally applied
4. Supports structured metadata

**Consequences**:
- (+) Cursor-native
- (+) Automatic context injection
- (-) Different from original spec

---

## Template for Future ADRs

```markdown
## ADR-XXX: [Title]

**Status**: Proposed | Accepted | Deprecated | Superseded

**Context**:
[What is the issue we're addressing?]

**Decision**:
[What did we decide?]

**Rationale**:
[Why did we make this decision?]

**Consequences**:
[What are the results? Both positive and negative.]
```
