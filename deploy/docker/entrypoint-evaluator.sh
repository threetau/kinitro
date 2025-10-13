#!/usr/bin/env bash
set -euo pipefail

if [[ "${RUN_MIGRATIONS:-1}" == "1" ]]; then
  echo "[evaluator] running validator database migrations..."
  /app/scripts/migrate_validator_db.sh
fi

echo "[evaluator] starting orchestrator..."
exec python -m evaluator.orchestrator --config /etc/kinitro/evaluator.toml "$@"
