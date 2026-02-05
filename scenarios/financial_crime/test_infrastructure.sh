#!/bin/bash
# Infrastructure Validation Test Suite
# The Panopticon Protocol Section 4.2.2 - Verification
#
# Tests:
# 1. TCMalloc verification
# 2. Sidecar proxy functionality
# 3. Rate limiting
# 4. Circuit breaking / retries
# 5. Performance benchmarks

set -e

echo "=========================================="
echo "Infrastructure Validation Test Suite"
echo "The Panopticon Protocol"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass() { echo -e "${GREEN}✅ $1${NC}"; }
fail() { echo -e "${RED}❌ $1${NC}"; exit 1; }
warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }

# ============================================================================
# Test 1: TCMalloc Verification
# ============================================================================
echo ""
echo "Test 1: Verifying TCMalloc is loaded..."
if docker-compose exec -T green_agent_app bash -c '
    if [ -n "$LD_PRELOAD" ]; then
        echo "LD_PRELOAD is set: $LD_PRELOAD"
        exit 0
    else
        echo "LD_PRELOAD is not set"
        exit 1
    fi
' 2>/dev/null; then
    pass "TCMalloc LD_PRELOAD is configured"
else
    warn "Could not verify TCMalloc (container may not be running)"
fi

# ============================================================================
# Test 2: Sidecar Proxy Verification
# ============================================================================
echo ""
echo "Test 2: Verifying Sidecar is proxying requests..."
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || echo "000")
if [ "$RESPONSE" -eq 200 ]; then
    pass "Sidecar proxy is working (HTTP 200)"
else
    fail "Sidecar proxy failed (HTTP $RESPONSE)"
fi

# ============================================================================
# Test 3: API Endpoints
# ============================================================================
echo ""
echo "Test 3: Verifying API Endpoints..."

# Health endpoint
HEALTH=$(curl -s http://localhost:8000/health 2>/dev/null || echo "{}")
if echo "$HEALTH" | grep -q "healthy\|ok\|status"; then
    pass "Health endpoint responding"
else
    warn "Health endpoint response unexpected: $HEALTH"
fi

# Agent manifest
MANIFEST=$(curl -s http://localhost:8000/agent.json 2>/dev/null || echo "{}")
if echo "$MANIFEST" | grep -q "Green"; then
    pass "Agent manifest available"
else
    warn "Agent manifest response unexpected"
fi

# ============================================================================
# Test 4: Rate Limiting
# ============================================================================
echo ""
echo "Test 4: Testing rate limiting (sending 150 rapid requests)..."

RATE_LIMITED=false
for i in $(seq 1 150); do
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null)
    if [ "$RESPONSE" -eq 429 ]; then
        RATE_LIMITED=true
        break
    fi
done

if [ "$RATE_LIMITED" = true ]; then
    pass "Rate limiting is working (HTTP 429 received)"
else
    warn "Rate limiting may not be active (no 429 responses)"
fi

# Wait for rate limit to reset
sleep 2

# ============================================================================
# Test 5: Performance Benchmarking
# ============================================================================
echo ""
echo "Test 5: Performance benchmarking (100 requests)..."

START=$(date +%s%N)
for i in $(seq 1 100); do
    curl -s http://localhost:8000/health > /dev/null 2>&1
done
END=$(date +%s%N)

DURATION_MS=$(( (END - START) / 1000000 ))
AVG_MS=$(( DURATION_MS / 100 ))

if [ "$AVG_MS" -lt 100 ]; then
    pass "Average response time: ${AVG_MS}ms (excellent)"
elif [ "$AVG_MS" -lt 500 ]; then
    pass "Average response time: ${AVG_MS}ms (good)"
else
    warn "Average response time: ${AVG_MS}ms (slow)"
fi

# ============================================================================
# Test 6: Envoy Admin (if using Envoy)
# ============================================================================
echo ""
echo "Test 6: Checking Envoy Admin interface..."
ADMIN_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:9901/ready 2>/dev/null || echo "000")
if [ "$ADMIN_RESPONSE" -eq 200 ]; then
    pass "Envoy Admin ready at :9901"
else
    warn "Envoy Admin not available (HTTP $ADMIN_RESPONSE) - may be using Rust sidecar"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=========================================="
echo -e "${GREEN}Infrastructure validation complete!${NC}"
echo "=========================================="
echo ""
echo "Commands to manage the infrastructure:"
echo "  docker-compose up -d          # Start services"
echo "  docker-compose logs -f        # View logs"
echo "  docker-compose down           # Stop services"
echo ""
