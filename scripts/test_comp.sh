#!/bin/bash

# Test script to create a competition using the admin API
# Requires an admin API key to be set in the ADMIN_API_KEY environment variable

# Check if API key is set
if [ -z "$ADMIN_API_KEY" ]; then
    echo "Error: ADMIN_API_KEY environment variable is not set"
    echo "Please set it with: export ADMIN_API_KEY=your_api_key_here"
    exit 1
fi

# Set default backend URL if not provided
BACKEND_URL=${BACKEND_URL:-"http://localhost:8080"}

echo "Creating test competition using admin API..."
echo "Backend URL: $BACKEND_URL"

curl -X POST "$BACKEND_URL/competitions" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ADMIN_API_KEY" \
  -d '{
    "name": "test",
    "description": "MetaWorld MT1 test competition",
    "benchmarks": [
        {
            "provider": "metaworld",
            "benchmark_name": "MT1",
            "config": {
                "env_name": "reach-v3",
                "episodes_per_task": 1,
                "max_episode_steps": 200
            }
        }
    ],
    "points": 50,
    "min_avg_reward": 0.0,
    "win_margin_pct": 0.05,
    "min_success_rate": 0.8
  }' \
  --verbose \
  --fail-with-body

# Check the exit status
if [ $? -eq 0 ]; then
    echo "Competition created successfully!"
else
    echo "Failed to create competition"
    exit 1
fi
