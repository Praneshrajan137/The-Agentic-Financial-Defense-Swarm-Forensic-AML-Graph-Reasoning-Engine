# Purple Agent -- Project Gamma

Autonomous forensic financial crime investigator.

**Competition:** AgentX-AgentBeats Finance Track
**Protocol:** The Panopticon Protocol v12.0

## Overview

Purple Agent receives investigation requests via A2A protocol, traverses
financial transaction graphs, detects money laundering patterns (structuring
and layering), synthesizes multi-modal evidence, and generates regulatory-
compliant SARs (FinCEN + FIU-IND).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run pre-flight checks
bash scripts/preflight.sh

# Start server
python src/main.py

# Run tests
pytest tests/ -v
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design.
