#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-deploy/docker/compose.yaml}"

echo "Pulling latest validator/evaluator images..."
docker compose -f "${COMPOSE_FILE}" pull validator evaluator

echo "Running database migrations..."
docker compose -f "${COMPOSE_FILE}" --profile ops run --rm migrator

echo "Restarting validator stack..."
if [[ "${USE_GPU_EVALUATOR:-0}" == "1" ]]; then
  docker compose -f "${COMPOSE_FILE}" up -d validator
  docker compose -f "${COMPOSE_FILE}" --profile gpu up -d evaluator-gpu
  docker compose -f "${COMPOSE_FILE}" stop evaluator || true
else
  docker compose -f "${COMPOSE_FILE}" up -d validator evaluator
  docker compose -f "${COMPOSE_FILE}" --profile gpu stop evaluator-gpu || true
fi

docker compose -f "${COMPOSE_FILE}" up -d watchtower

echo "Update complete."
