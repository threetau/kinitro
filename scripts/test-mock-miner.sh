#!/usr/bin/env bash
# Test mock miner endpoints
# Run this after starting the mock miner with: uv run kinitro mock-miner --port 8001

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load environment if available
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi

MINER_PORT="${MOCK_MINER_PORT:-8001}"
MINER_URL="http://localhost:$MINER_PORT"

echo "=== Testing Mock Miner at $MINER_URL ==="
echo ""

# Track results
PASSED=0
FAILED=0

test_endpoint() {
    local name="$1"
    local method="$2"
    local endpoint="$3"
    local data="${4:-}"

    echo -n "Testing $name... "

    local args=(-s -w "\n%{http_code}" -X "$method" "$MINER_URL$endpoint")
    if [[ -n "$data" ]]; then
        args+=(-H "Content-Type: application/json" -d "$data")
    fi

    local response
    response=$(curl "${args[@]}" 2>/dev/null) || {
        echo "FAILED (connection error)"
        ((FAILED++))
        return 1
    }

    local http_code
    http_code=$(echo "$response" | tail -n1)
    local body
    body=$(echo "$response" | sed '$d')

    if [[ "$http_code" =~ ^2[0-9][0-9]$ ]]; then
        echo "OK ($http_code)"
        echo "  Response: $body"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo "FAILED ($http_code)"
        echo "  Response: $body"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# Test health endpoint
test_endpoint "health" "GET" "/health"

# Test reset endpoint
test_endpoint "reset" "POST" "/reset" '{
  "task_config": {
    "env_id": "metaworld/pick-place-v3",
    "seed": 42
  }
}'

# Test act endpoint with proprioceptive observation
test_endpoint "act" "POST" "/act" '{
  "obs": {
    "proprio": {
      "ee_pos": [0.0, 0.5, 0.2],
      "ee_quat": [0.0, 0.0, 0.0, 1.0],
      "ee_vel_lin": [0.0, 0.0, 0.0],
      "ee_vel_ang": [0.0, 0.0, 0.0],
      "gripper": [1.0]
    },
    "rgb": {}
  }
}'

echo ""
echo "=== Results ==="
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""

if [[ $FAILED -gt 0 ]]; then
    echo "Some tests failed. Is the mock miner running?"
    echo "  Start it with: uv run kinitro mock-miner --port $MINER_PORT"
    exit 1
else
    echo "All tests passed!"
    exit 0
fi
