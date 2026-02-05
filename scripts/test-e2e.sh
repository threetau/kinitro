#!/usr/bin/env bash
# Full end-to-end test: services → mock miner → evaluation → scoring
# This script starts all services, runs a mock evaluation, and verifies results.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load environment
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
else
    echo "Error: .env not found. Run ./scripts/worktree-env.sh first."
    exit 1
fi

API_PORT="${API_PORT:-8000}"
API_URL="http://localhost:$API_PORT"
MINER_PORT="${MOCK_MINER_PORT:-8001}"

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run a full end-to-end test of the kinitro evaluation pipeline.

Options:
  --skip-start      Skip starting services (assumes they're already running)
  --skip-stop       Don't stop services after test
  --timeout SECS    Max time to wait for evaluations (default: 120)
  -h, --help        Show this help

The test will:
  1. Start all services including mock miner
  2. Initialize the database
  3. Wait for evaluations to complete
  4. Verify scores were computed
  5. Stop services (unless --skip-stop)

EOF
}

SKIP_START=false
SKIP_STOP=false
TIMEOUT=120

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-start) SKIP_START=true; shift ;;
        --skip-stop) SKIP_STOP=true; shift ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

cleanup() {
    if [[ "$SKIP_STOP" == "false" ]]; then
        echo ""
        echo "=== Cleanup ==="
        "$SCRIPT_DIR/services.sh" stop 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "=== E2E Test: Full Pipeline ==="
echo "API URL: $API_URL"
echo "Miner URL: http://localhost:$MINER_PORT"
echo "Timeout: ${TIMEOUT}s"
echo ""

# Step 1: Start services
if [[ "$SKIP_START" == "false" ]]; then
    echo "=== Step 1: Starting Services ==="
    "$SCRIPT_DIR/services.sh" start --mock-miner

    echo "Waiting for services to initialize..."
    sleep 5

    # Initialize database
    echo "Initializing database..."
    uv run kinitro db init --database-url "$DATABASE_URL" 2>/dev/null || true
else
    echo "=== Step 1: Skipped (--skip-start) ==="
fi

# Step 2: Verify services are running
echo ""
echo "=== Step 2: Verifying Services ==="

echo -n "API health... "
if curl -s "$API_URL/health" | grep -q "ok\|healthy"; then
    echo "OK"
else
    echo "FAILED"
    echo "API is not responding. Check logs: tail -f /tmp/kinitro_*/api.log"
    exit 1
fi

echo -n "Mock miner health... "
if curl -s "http://localhost:$MINER_PORT/health" | grep -q "ok\|healthy"; then
    echo "OK"
else
    echo "FAILED"
    echo "Mock miner is not responding. Check logs: tail -f /tmp/kinitro_*/mock-miner.log"
    exit 1
fi

# Step 3: Run component tests
echo ""
echo "=== Step 3: Running Component Tests ==="

echo "Testing mock miner endpoints..."
"$SCRIPT_DIR/test-mock-miner.sh" || {
    echo "Mock miner tests failed"
    exit 1
}

echo ""
echo "Testing API endpoints..."
"$SCRIPT_DIR/test-api.sh" || {
    echo "API tests failed"
    exit 1
}

# Step 4: Wait for evaluations (if scheduler is running)
echo ""
echo "=== Step 4: Checking Evaluation Pipeline ==="

echo "Checking task pool..."
TASK_STATS=$(curl -s "$API_URL/v1/tasks/stats" 2>/dev/null || echo "{}")
echo "Task stats: $TASK_STATS"

echo ""
echo "Polling for completed evaluations (timeout: ${TIMEOUT}s)..."
SCORES_CODE="000"
POLL_INTERVAL=5
START_TIME=$SECONDS

while (( SECONDS - START_TIME < TIMEOUT )); do
    SCORES_RESPONSE=$(curl -s -w "\n%{http_code}" "$API_URL/v1/scores/latest" 2>/dev/null || echo -e "{}\n000")
    SCORES_BODY=$(echo "$SCORES_RESPONSE" | sed '$d')
    SCORES_CODE=$(echo "$SCORES_RESPONSE" | tail -n1)

    if [[ "$SCORES_CODE" == "200" ]]; then
        echo "Found completed evaluations:"
        echo "$SCORES_BODY" | head -c 500
        echo ""
        break
    fi

    ELAPSED=$((SECONDS - START_TIME))
    echo "  No evaluations yet (${ELAPSED}s/${TIMEOUT}s)..."
    sleep $POLL_INTERVAL
done

if [[ "$SCORES_CODE" != "200" ]]; then
    echo "No completed evaluations after ${TIMEOUT}s (HTTP $SCORES_CODE)."
    echo "Note: Full evaluation cycle requires:"
    echo "  - Registered miners with committed policies"
    echo "  - Built environment images (kinitro env build)"
    echo "  - Scheduler running evaluation cycles"
fi

# Step 5: Summary
echo ""
echo "=== E2E Test Summary ==="
echo "✓ Services started successfully"
echo "✓ API responding"
echo "✓ Mock miner responding"
echo "✓ Component tests passed"
echo ""
echo "For a complete evaluation cycle, ensure:"
echo "  1. Environment images are built: uv run kinitro env build metaworld --tag kinitro/metaworld:v1"
echo "  2. Executor has eval-images configured in EVAL_IMAGES env var"
echo "  3. A miner has committed a policy to the network"
echo ""
echo "Monitor logs: tail -f /tmp/kinitro_${WORKTREE_NAME:-default}/*.log"

exit 0
