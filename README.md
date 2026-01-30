# The Panopticon Protocol

**A Zero-Failure Synthetic Financial Crime Investigation Benchmark**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Abstract

The **Green Financial Crime Agent** generates synthetic financial transaction networks with surgically injected money laundering patterns (structuring and layering). It evaluates Purple Agents on their ability to:

1. **Pattern Identification** - Detect structuring (smurfing) and layering typologies
2. **Evidence Analysis** - Extract clues from unstructured text (SARs, emails)
3. **Cognitive Reasoning** - Distinguish reliable from conflicting evidence
4. **Efficiency** - Minimize tool calls while maximizing detection accuracy

The benchmark supports 10 difficulty levels, from trivial (4-hour time windows, clustered amounts) to expert (3-month spreads, minimal patterns, decoy transactions).

### Key Innovations

| Feature | Description |
|---------|-------------|
| **Statistical Realism** | SDV Gaussian Copulas for correlated transaction data |
| **Locale-Aligned Entities** | SWIFT/IBAN codes match country jurisdictions |
| **Sherlock Holmes Mode** | Unstructured evidence (SARs, emails) requiring NLU |
| **Efficiency Metrics** | Tool call tracking alongside accuracy |
| **Dynamic Difficulty** | 10 levels from trivial to expert |
| **Conflicting Evidence** | Tests hallucination resistance |

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.10+ (for local development)
- 4GB RAM minimum

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/financial-crime-benchmark.git
cd financial-crime-benchmark

# Build and run both agents
docker-compose up --build

# The Green Agent will:
# 1. Generate synthetic financial data (seed=42, difficulty=5)
# 2. Start the A2A server on port 8000
# 3. Wait for Purple Agent connections

# In a separate terminal, run the baseline Purple Agent
docker-compose run purple-agent
```

### Option 2: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data (difficulty 5, seed 42)
python main.py generate --output-dir ./outputs --seed 42 --difficulty 5

# Start A2A server
python main.py serve --port 8000

# In another terminal, run baseline Purple Agent
cd purple_agent
pip install -r requirements.txt
python src/baseline_agent.py --green-url http://localhost:8000 --nodes 0 42 100
```

### Option 3: Single Command End-to-End

```bash
# Run benchmark with reproducibility demonstration
python scripts/run_benchmark.py --seed 42 --difficulty 5 --runs 3 --output results.json
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     THE PANOPTICON PROTOCOL                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐         ┌──────────────────────────┐  │
│  │    GREEN AGENT       │         │      PURPLE AGENT        │  │
│  │   (The Judge)        │  A2A    │    (The Investigator)    │  │
│  │                      │ ◄─────► │                          │  │
│  │  - Graph Generator   │  HTTP   │  - Detection Heuristics  │  │
│  │  - Crime Injector    │  JSON   │  - Evidence Analysis     │  │
│  │  - Evidence Gen      │         │  - Investigation Report  │  │
│  │  - Assessment API    │         │                          │  │
│  └──────────────────────┘         └──────────────────────────┘  │
│           │                                    │                 │
│           ▼                                    ▼                 │
│  ┌──────────────────────┐         ┌──────────────────────────┐  │
│  │   Ground Truth       │         │    Assessment Score      │  │
│  │   - Crime patterns   │         │    - Pattern ID: 28%     │  │
│  │   - Entity IDs       │         │    - Evidence: 20%       │  │
│  │   - Evidence docs    │         │    - Narrative: 16%      │  │
│  └──────────────────────┘         │    - Completeness: 16%   │  │
│                                   │    - Efficiency: 20%     │  │
│                                   └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Green Agent Components

| Module | Purpose | Key Technology |
|--------|---------|----------------|
| `graph_generator.py` | Scale-free financial networks | NetworkX, Faker, SDV |
| `crime_injector.py` | Structuring & layering patterns | Difficulty-based obfuscation |
| `evidence_generator.py` | SARs, emails, conflicting docs | NLU challenge generation |
| `a2a_interface.py` | HTTP API for Purple Agents | FastAPI, Protobuf |

### Purple Agent (Baseline)

The included baseline agent demonstrates the A2A protocol with simple heuristics:
- **Structuring Detection**: Fan-in pattern (15+ senders, $9k-$9.8k amounts)
- **Layering Detection**: Chain transfers with decay patterns
- **Performance**: ~60% detection rate on difficulty 5

---

## API Reference

All endpoints use JSON. Include `X-Participant-ID` header for efficiency tracking.

### Discovery

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/agent.json` | GET | Agent manifest |

### Investigation Tools

#### Get Transactions
```http
POST /a2a/tools/get_transactions
Content-Type: application/json
X-Participant-ID: my-agent-id

{
  "account_id": "42",
  "limit": 100,
  "direction": "both"
}
```

#### Get KYC Profile
```http
POST /a2a/tools/get_kyc_profile
Content-Type: application/json

{
  "account_id": "42"
}
```

#### Get Evidence Documents
```http
POST /a2a/tools/get_evidence
Content-Type: application/json

{
  "contains_keyword": "structuring",
  "limit": 10
}
```

#### Get Account Connections
```http
POST /a2a/tools/get_account_connections
Content-Type: application/json

{
  "account_id": "42",
  "depth": 1
}
```

### Assessment

#### Submit Investigation
```http
POST /a2a/investigation_assessment
Content-Type: application/json

{
  "participant_id": "my-agent-id",
  "investigation_data": {
    "identified_crimes": [
      {"crime_type": "structuring", "nodes": ["42"]}
    ],
    "suspicious_accounts": ["42", "100"],
    "narrative": "Investigation summary...",
    "transaction_ids": ["txn_abc123"],
    "temporal_patterns": true,
    "amount_patterns": true
  }
}
```

**Response:**
```json
{
  "score": 75.5,
  "feedback": "Good investigation with solid pattern identification.",
  "rubric_breakdown": {
    "pattern_identification": 80.0,
    "evidence_quality": 75.0,
    "narrative_clarity": 70.0,
    "completeness": 65.0
  },
  "tool_call_count": 45,
  "efficiency_score": 100.0,
  "efficiency_rank": "excellent"
}
```

---

## Evaluation Rubric

Purple Agents are scored on five dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **Pattern Identification** | 28% | Precision/Recall/F1 for crime detection |
| **Evidence Quality** | 20% | Use of transaction data and supporting evidence |
| **Narrative Clarity** | 16% | Quality and structure of investigation report |
| **Completeness** | 16% | Coverage of ground truth indicators |
| **Efficiency** | 20% | Tool calls used (fewer = better) |

### Efficiency Scoring

| Tool Calls | Score | Rank |
|------------|-------|------|
| 10-50 | 100 | Excellent |
| 51-100 | 80 | Good |
| 101-200 | 60 | Fair |
| 200+ | <40 | Poor |

---

## Difficulty Levels

| Level | Time Spread | Amount Variance | Detection Rate (Baseline) |
|-------|-------------|-----------------|---------------------------|
| 1-3 | 4 hours | $9,500-$9,700 | 95%+ |
| 4-6 | 48 hours | $9,000-$9,800 | 60-70% |
| 7-8 | 1 week | $7,500-$9,800 + decoys | 30-40% |
| 9-10 | 3 months | Minimal patterns, long gaps | <20% |

### Expert Mode (9-10) Includes:
- Transactions mixed with legitimate activity
- Long time gaps between crime transactions
- Minimal pattern clustering
- Decoy transactions from crime sources
- Conflicting evidence documents

---

## Reproducibility

This benchmark is designed for reproducible evaluation.

### Determinism Guarantees

1. **Seeded Random Generation**: All random operations use configurable seed
2. **Deterministic Crime Injection**: Same seed = identical crime patterns
3. **Fixed Assessment Logic**: Rule-based scoring (no LLM variance)
4. **Documented Configuration**: All parameters in `scenario.toml`

### Running Reproducibility Tests

```bash
# Run 3 evaluations with identical configuration
python scripts/run_benchmark.py --seed 42 --difficulty 5 --runs 3 --output results.json

# Verify results
cat results.json
```

**Expected Output:**
```json
{
  "seed": 42,
  "difficulty": 5,
  "runs": [
    {"run": 1, "score": 62.5, "tool_calls": 23},
    {"run": 2, "score": 62.5, "tool_calls": 23},
    {"run": 3, "score": 62.5, "tool_calls": 23}
  ],
  "variance": {
    "score": 0.0,
    "tool_calls": 0.0
  },
  "reproducible": true
}
```

### Configuration File

See [`scenarios/financial_crime/scenario.toml`](scenarios/financial_crime/scenario.toml) for all configurable parameters:

```toml
[scenario]
name = "financial_crime_investigation"
default_difficulty = 5

[assessment.scoring_weights]
pattern_identification = 0.28
evidence_quality = 0.20
narrative_clarity = 0.16
completeness = 0.16
efficiency = 0.20

[configuration]
graph_size = 1000
crime_types = ["structuring", "layering"]
```

---

## Project Structure

```
.
├── main.py                    # CLI entry point
├── Dockerfile                 # Green Agent container
├── docker-compose.yml         # Multi-agent orchestration
├── requirements.txt           # Python dependencies
├── src/
│   ├── core/
│   │   ├── graph_generator.py    # NetworkX + SDV
│   │   ├── crime_injector.py     # Structuring/Layering
│   │   ├── evidence_generator.py # SARs/Emails
│   │   ├── a2a_interface.py      # FastAPI + Protobuf
│   │   └── sdv_models.py         # Gaussian Copulas
│   └── utils/
│       └── validators.py
├── purple_agent/              # Baseline investigator
│   ├── src/
│   │   └── baseline_agent.py
│   ├── Dockerfile
│   └── requirements.txt
├── scripts/
│   ├── run_benchmark.py       # Reproducibility demonstration
│   └── init_server.py         # Deterministic startup
├── tests/                     # Test suite
├── scenarios/
│   └── financial_crime/
│       ├── scenario.toml      # Configuration
│       └── README.md          # Detailed documentation
└── outputs/                   # Generated data (gitignored)
```

---

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/ -v
```

---

## Resource Requirements

| Resource | Requirement |
|----------|-------------|
| RAM | 4GB minimum, 8GB recommended |
| Disk | 500MB for dependencies + outputs |
| CPU | Any modern CPU (graph generation is fast) |
| Network | Required for Docker pulls |

### Performance Metrics

- **Graph Generation**: 1,000 nodes in <10 seconds
- **Memory Usage**: <2GB for full pipeline
- **API Response**: <100ms for most endpoints
- **Protobuf**: 80% smaller than JSON responses

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest`
4. Submit a pull request

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{panopticon_protocol,
  title = {The Panopticon Protocol: A Synthetic Financial Crime Investigation Benchmark},
  year = {2026},
  url = {https://github.com/YOUR_USERNAME/financial-crime-benchmark}
}
```

---

## Contact

- **Competition**: AgentX-AgentBeats Financial Crime Track
- **Version**: 1.0.0
- **Status**: Production Ready
