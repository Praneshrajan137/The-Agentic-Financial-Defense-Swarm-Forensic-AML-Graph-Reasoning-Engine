#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Pre-flight Validation Script -- Purple Agent v7.0
# Run this BEFORE deployment to verify all prerequisites are met.
# Exit code 0 = all checks passed. Non-zero = failure.
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

ERRORS=0
WARNINGS=0

echo "=== Purple Agent Pre-flight Check v7.0 ==="
echo ""

# 1. Python version
echo -n "[CHECK] Python 3.11+ ... "
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if python3 -c 'import sys; assert sys.version_info >= (3, 11)' 2>/dev/null; then
    echo "OK ($PYVER)"
else
    echo "FAIL ($PYVER < 3.11)"
    ERRORS=$((ERRORS + 1))
fi

# 2. Dependencies installed
echo -n "[CHECK] Core dependencies ... "
if python3 -c "import networkx, langgraph, httpx, fastapi, spacy, google.protobuf" 2>/dev/null; then
    echo "OK"
else
    echo "FAIL (run: pip install -r requirements.txt)"
    ERRORS=$((ERRORS + 1))
fi

# 3. spaCy model (respects SPACY_MODEL env var)
SPACY_MODEL_TO_CHECK=${SPACY_MODEL:-en_core_web_sm}
echo -n "[CHECK] spaCy model ($SPACY_MODEL_TO_CHECK) ... "
if python3 -c "import spacy; spacy.load('$SPACY_MODEL_TO_CHECK')" 2>/dev/null; then
    echo "OK"
else
    echo "FAIL (run: python -m spacy download $SPACY_MODEL_TO_CHECK)"
    ERRORS=$((ERRORS + 1))
fi

# 4. Protobuf bindings
echo -n "[CHECK] Protobuf bindings ... "
if python3 -c "from protos import financial_crime_pb2" 2>/dev/null; then
    echo "OK"
else
    echo "FAIL (run: python -m grpc_tools.protoc -I protos --python_out=protos protos/financial_crime.proto)"
    ERRORS=$((ERRORS + 1))
fi

# 5. Environment
echo -n "[CHECK] .env file or OPENAI_API_KEY ... "
if [ -f ".env" ] || [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "OK"
else
    echo "WARN (no .env file and OPENAI_API_KEY not set -- LLM calls will fail)"
    WARNINGS=$((WARNINGS + 1))
fi

# 6. Config imports
echo -n "[CHECK] Config module ... "
if python3 -c "from src.config import STRUCTURING_MIN_AMOUNT_USD, SAR_LLM_SEED, STRUCTURING_MIN_COUNT_INR; print('OK')" 2>/dev/null; then
    echo "OK"
else
    echo "FAIL (src/config.py import error -- are ALN-02 constants present?)"
    ERRORS=$((ERRORS + 1))
fi

# 7. PYTHONHASHSEED
echo -n "[CHECK] PYTHONHASHSEED=0 (determinism) ... "
if [ "${PYTHONHASHSEED:-unset}" = "0" ]; then
    echo "OK"
else
    echo "WARN (set PYTHONHASHSEED=0 for deterministic outputs)"
    WARNINGS=$((WARNINGS + 1))
fi

# 8. Tests runnable
echo -n "[CHECK] pytest discoverable ... "
if python3 -m pytest --collect-only tests/ -q 2>/dev/null | grep -q "test"; then
    echo "OK"
else
    echo "WARN (no tests collected -- have you created test files?)"
    WARNINGS=$((WARNINGS + 1))
fi

# 9. Docker available (for build verification)
echo -n "[CHECK] Docker available ... "
if command -v docker &>/dev/null; then
    echo "OK ($(docker --version 2>/dev/null | head -c 40))"
else
    echo "WARN (Docker not found -- needed for D2: Docker Build Verification)"
    WARNINGS=$((WARNINGS + 1))
fi

echo ""
if [ $ERRORS -gt 0 ]; then
    echo "=== PRE-FLIGHT FAILED: $ERRORS error(s), $WARNINGS warning(s) ==="
    exit 1
else
    echo "=== PRE-FLIGHT PASSED ($WARNINGS warning(s)) ==="
    exit 0
fi
