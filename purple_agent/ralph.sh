#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Ralph Wiggum Loop v12.0 -- Die-and-restart for context rot prevention
#
# v10.0 [CRIT-08 FIX]: cd to project root, not scripts/.
# v9.0  [CRIT-06 FIX]: STATE_DIR re-added.
# v7.0  [BUG-01 FIX]:  No set -e in iteration loop.
# ═══════════════════════════════════════════════════════════════════
set -uo pipefail

# Navigate to PROJECT ROOT (parent of scripts/).
# ralph.sh lives in scripts/. "$(dirname "$0")" resolves to "scripts".
cd "$(dirname "$0")/.."

CHILD_PID=0

cleanup() {
    echo "=== Received shutdown signal. Cleaning up... ==="
    if [ "$CHILD_PID" -ne 0 ]; then
        kill -TERM "$CHILD_PID" 2>/dev/null
        wait "$CHILD_PID" 2>/dev/null
    fi
    exit 0
}
trap cleanup SIGTERM SIGINT

MAX_ITERATIONS=${1:-25}
TASK_TIMEOUT=${2:-120}
SLEEP_SECONDS=${3:-2}

STATE_DIR="./state"
mkdir -p "$STATE_DIR"

echo "=== Ralph Wiggum Loop v12.0: Max $MAX_ITERATIONS iterations, ${TASK_TIMEOUT}s timeout, ${SLEEP_SECONDS}s sleep ==="
echo "=== Working directory: $(pwd) ==="

for i in $(seq 1 "$MAX_ITERATIONS"); do
    echo ""
    echo "=== Ralph iteration $i / $MAX_ITERATIONS ==="
    echo ""

    timeout "${TASK_TIMEOUT}s" python -m src.ralph_runner \
        --state-dir "$STATE_DIR" \
        --prd plans/prd.json \
        --progress progress.txt &
    CHILD_PID=$!
    wait "$CHILD_PID" && EXIT_CODE=0 || EXIT_CODE=$?
    CHILD_PID=0

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "=== ALL TASKS COMPLETE ==="
        exit 0
    elif [ $EXIT_CODE -eq 124 ]; then
        echo "WARNING: Task timed out after ${TASK_TIMEOUT}s. Restarting..."
    else
        echo "--- Iteration $i finished (exit code: $EXIT_CODE). Restarting... ---"
    fi

    sleep "$SLEEP_SECONDS"
done

echo "WARNING: Reached max iterations ($MAX_ITERATIONS) without completion."
exit 1
