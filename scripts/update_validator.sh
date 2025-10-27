#!/usr/bin/env bash
set -euo pipefail

DEFAULT_ARGS=(-f deploy/docker/compose.base.yaml)
if [[ -n "${COMPOSE_FILES:-}" ]]; then
  # shellcheck disable=SC2206
  COMPOSE_ARGS=(${COMPOSE_FILES})
else
  COMPOSE_ARGS=("${DEFAULT_ARGS[@]}")
fi

echo "Pulling latest validator/evaluator images..."
docker compose "${COMPOSE_ARGS[@]}" pull validator evaluator

echo "Running database migrations..."
docker compose "${COMPOSE_ARGS[@]}" --profile ops run --rm migrator

echo "Restarting validator stack..."
if [[ "${USE_GPU_EVALUATOR:-0}" == "1" ]]; then
  docker compose "${COMPOSE_ARGS[@]}" up -d validator
  docker compose "${COMPOSE_ARGS[@]}" --profile gpu up -d evaluator-gpu
  docker compose "${COMPOSE_ARGS[@]}" stop evaluator || true
else
  docker compose "${COMPOSE_ARGS[@]}" up -d validator evaluator
  docker compose "${COMPOSE_ARGS[@]}" --profile gpu stop evaluator-gpu || true
fi

docker compose "${COMPOSE_ARGS[@]}" up -d watchtower

echo "Update complete."
