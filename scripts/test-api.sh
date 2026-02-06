#!/usr/bin/env bash
# Test API endpoints
# Run this after starting the API with: ./scripts/services.sh start --api-only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load environment if available
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi

API_PORT="${API_PORT:-8000}"
API_URL="http://localhost:$API_PORT"

echo "=== Testing API at $API_URL ==="
echo ""

# Track results
PASSED=0
FAILED=0

test_endpoint() {
    local name="$1"
    local endpoint="$2"
    local expected_status="${3:-200}"  # Can be "200" or "200|404" for multiple valid codes

    echo -n "Testing $name... "

    local response
    response=$(curl -s -w "\n%{http_code}" "$API_URL$endpoint" 2>/dev/null) || {
        echo "FAILED (connection error)"
        FAILED=$((FAILED + 1))
        return 1
    }

    local http_code
    http_code=$(echo "$response" | tail -n1)
    local body
    body=$(echo "$response" | sed '$d')

    # Check if http_code matches any of the expected statuses (pipe-separated)
    if [[ "|$expected_status|" == *"|$http_code|"* ]]; then
        echo "OK ($http_code)"
        # Truncate long responses
        if [[ ${#body} -gt 200 ]]; then
            echo "  Response: ${body:0:200}..."
        else
            echo "  Response: $body"
        fi
        PASSED=$((PASSED + 1))
        return 0
    else
        echo "FAILED (expected $expected_status, got $http_code)"
        echo "  Response: $body"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# Run all tests (disable exit-on-error so we get the full summary)
set +e

test_endpoint "health" "/health"
test_endpoint "miners list" "/v1/miners"
test_endpoint "environments list" "/v1/environments"

# Scores endpoints (may return 404 if no evaluations yet)
test_endpoint "latest scores" "/v1/scores/latest" "200|404"
test_endpoint "latest weights" "/v1/weights/latest" "200|404"

# Task pool stats
test_endpoint "task stats" "/v1/tasks/stats"

set -e

echo ""
echo "=== Results ==="
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""

if [[ $FAILED -gt 0 ]]; then
    echo "Some tests failed. Is the API running?"
    echo "  Start it with: ./scripts/services.sh start --api-only"
    exit 1
else
    echo "All tests passed!"
    exit 0
fi
