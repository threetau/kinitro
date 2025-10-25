#!/usr/bin/env bash
set -euo pipefail

if [[ "${RUN_MIGRATIONS:-1}" == "1" ]]; then
  echo "[backend] running database migrations..."
  /app/scripts/migrate_backend_db.sh
fi

echo "[backend] starting API server..."
exec python -m backend.__main__ --config /etc/kinitro/backend.toml "$@"
