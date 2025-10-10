#!/usr/bin/env bash
set -euo pipefail

# Allow overriding the command entirely.
if [[ "${1:-}" == "--" ]]; then
  shift
fi

echo "[validator] running database migrations..."
/app/scripts/migrate_validator_db.sh

echo "[validator] launching websocket validator..."
exec python -m validator --config /etc/kinitro/validator.toml "$@"
