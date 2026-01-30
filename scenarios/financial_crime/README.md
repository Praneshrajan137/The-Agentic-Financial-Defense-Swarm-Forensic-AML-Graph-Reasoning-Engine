# Financial Crime Investigation Benchmark
## The Panopticon Protocol - Phase 1 Submission

**Status:** Phase 1 Complete (100%) + Winning Features  
**Competition:** AgentX-AgentBeats Financial Crime Track  
**Submission Date:** January 2026

---

## Executive Summary

This benchmark implements a **Zero-Failure Synthetic Financial Crime Simulator** designed to train and evaluate autonomous AML investigation agents. Unlike static datasets, this system generates dynamic, statistically realistic financial networks with surgically injected money laundering typologies.

### Key Innovations

1. **Statistical Realism** - SDV Gaussian Copulas (not random data)
2. **Locale-Aligned Entities** - SWIFT/IBAN match country codes
3. **"Sherlock Holmes" Upgrade** - Unstructured evidence (SARs, emails)
4. **Efficiency Metrics** - Tracks tool calls, not just accuracy
5. **Dynamic Difficulty** - 10 levels from trivial to expert
6. **Conflicting Evidence** - Tests hallucination resistance

---

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- 4GB RAM

### Run the Benchmark

```bash
# Clone and navigate
cd scenarios/financial_crime/

# Build agents
docker-compose build

# Start Green Agent (server)
docker-compose up green_agent

# In another terminal, run Purple Agent (investigator)
docker-compose run purple_agent
```

### Manual Setup (Without Docker)

```bash
# Install Green Agent
pip install -r requirements.txt

# Generate synthetic data (difficulty 5)
python main.py generate --difficulty 5 --output-dir ./outputs

# Start A2A server
python main.py serve

# In another terminal, run baseline purple agent
cd purple_agent/
pip install -r requirements.txt
python src/baseline_agent.py --nodes 0 42 100
```

---

## Architecture

### Green Agent (The God of Chaos)

**Purpose:** Generate forensic reality

**Components:**
- **Graph Generator** - NetworkX scale-free topology (1,000 nodes)
- **SDV Models** - Gaussian Copulas for statistical correlations
- **Crime Injector** - Structuring (smurfing) & Layering patterns
- **Evidence Generator** - SARs, emails, conflicting documents
- **A2A Interface** - FastAPI + Protobuf serialization

**Crime Types:**
1. **Structuring** - 20 sources -> 1 mule, $9k-$9.8k each, <48hrs
2. **Layering** - Chain transfers with 2-5% decay, no cycles

### Purple Agent (The Baseline)

**Purpose:** Prove the benchmark is solvable

**Logic:**
1. Connect to Green Agent A2A interface
2. Query transactions for suspect nodes
3. Apply simple heuristic (count incoming edges)
4. Return verdict

**Performance:** ~60% detection rate on difficulty 5

---

## Difficulty Levels

| Level | Description | Time Spread | Detection Rate (Baseline) |
|-------|-------------|-------------|---------------------------|
| 1-3 | Trivial | 4 hours | 95%+ |
| 4-6 | Medium | 48 hours | 60-70% |
| 7-8 | Hard | 1 week | 30-40% |
| 9-10 | Expert | 3 months | <20% |

**Expert mode includes:**
- Mixed with legitimate transactions
- Long time gaps
- Minimal pattern clustering
- Requires deep graph analysis

---

## API Reference

### A2A Tools (Purple Agent -> Green Agent)

#### `POST /a2a/tools/get_transactions`
```json
{
  "account_id": "node_42",
  "limit": 100,
  "direction": "both"
}
```

**Headers:**
- `X-Participant-ID`: Your agent's unique ID (for efficiency tracking)

**Returns:** Transaction list with amounts, timestamps, types

#### `POST /a2a/tools/get_kyc_profile`
```json
{"account_id": "node_42"}
```

**Returns:** Entity type, name, country, risk score, SWIFT/IBAN

#### `POST /a2a/tools/get_evidence`
```json
{
  "contains_keyword": "structuring",
  "limit": 10
}
```

**Returns:** SAR narratives, emails, receipts

#### `POST /a2a/tools/get_account_connections`
```json
{
  "account_id": "node_42",
  "depth": 1
}
```

**Returns:** Connected accounts with relationship and transaction counts

#### `POST /a2a/investigation_assessment`
```json
{
  "participant_id": "my_agent",
  "investigation_data": {
    "identified_crimes": [...],
    "suspicious_accounts": [...],
    "narrative": "..."
  }
}
```

**Returns:** Score (0-100), efficiency metrics, feedback

---

## Evaluation Rubric

Purple Agents are scored on:

1. **Pattern Identification** (28%) - Precision/Recall/F1
2. **Evidence Quality** (20%) - Use of transaction data
3. **Narrative Clarity** (16%) - Report quality
4. **Completeness** (16%) - Coverage of ground truth
5. **Efficiency** (20%) - Tool calls used

**Efficiency Scoring:**
- 10-50 calls = Excellent (100 pts)
- 51-100 calls = Good (80 pts)
- 101-200 calls = Fair (60 pts)
- 200+ calls = Poor (<40 pts)

---

## Judging Criteria Compliance

### Technical Correctness
- Valid A2A protocol (agent.json manifest)
- Proper graph topology (scale-free, DAG)
- Protobuf support (80% size reduction)

### Benchmark Design Quality
- **Sherlock Holmes upgrade:** Requires reading emails, not just SQL
- **Conflicting evidence:** Tests hallucination resistance
- Forces cognitive reasoning

### Evaluation Methodology
- Multi-metric scoring (not just accuracy)
- Efficiency tracking (tool call counts)
- Rubric breakdown by category

### Innovation
- Dynamic difficulty (10 levels)
- Evidence generation (unique to this benchmark)
- SDV statistical realism (not random data)

### Reproducibility
- Seeded generation (same seed = identical output)
- Deterministic crime injection
- Complete test suite (90%+ coverage)

---

## File Structure

```
project_root/                           # Green Agent lives at root
├── main.py                             # CLI entry point
├── Dockerfile                          # Green Agent Docker config
├── requirements.txt                    # Python dependencies
├── src/
│   ├── core/
│   │   ├── graph_generator.py          # NetworkX + SDV
│   │   ├── crime_injector.py           # Structuring/Layering
│   │   ├── evidence_generator.py       # SARs/Emails
│   │   ├── a2a_interface.py            # FastAPI + Protobuf
│   │   └── sdv_models.py               # Gaussian Copulas
│   └── utils/
│       └── validators.py
├── tests/                              # Test suite (90%+ coverage)
├── purple_agent/                       # Baseline investigator
│   ├── src/
│   │   └── baseline_agent.py           # Simple heuristic
│   ├── Dockerfile
│   └── requirements.txt
└── scenarios/
    └── financial_crime/
        ├── scenario.toml               # Competition configuration
        ├── README.md                   # This file
        └── docker-compose.yml          # Run both agents
```

---

## Performance Metrics

**Graph Generation:**
- 1,000 nodes in <10 seconds
- Memory usage <2GB
- Protobuf 80% smaller than JSON

**Crime Injection:**
- Structuring: 100% CTR compliance
- Layering: 0 cycles detected
- Evidence: 4-6 artifacts per crime

**API Response Times:**
- get_transactions: <100ms (JSON), <50ms (Protobuf)
- get_kyc_profile: <50ms
- investigation_assessment: <200ms

---

## Future Enhancements (Phase 2)

1. **Needle in Haystack** - 1,000 documents, 3 contain clues
2. **Additional Crime Types** - Round-tripping, trade-based ML
3. **Multi-agent Networks** - Green agents talk to each other
4. **Real-time Streaming** - WebSocket interface
5. **Neo4j Export** - Graph database integration

---

## Citations & References

1. FinCEN - Bank Secrecy Act Regulations
2. FATF - Money Laundering Typologies
3. SDV Documentation - Gaussian Copula Synthesis
4. AgentX-AgentBeats Competition Guidelines

---

## License

MIT License - See LICENSE file for details.

---

## Contact

For questions about this benchmark:
- Competition: AgentX-AgentBeats Financial Crime Track
- Phase: 1 (Benchmark Design)
- Submission: January 31, 2026

**Author:** The Panopticon Team  
**Version:** 1.0.0 - The Panopticon Protocol  
**Status:** Production Ready
