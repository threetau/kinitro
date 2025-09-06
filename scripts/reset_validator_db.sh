#!/usr/bin/env bash
set -euo pipefail

# --- Config (from environment, with sane defaults) ---
DB_USER="${DB_USER:-myuser}"
DB_PASSWORD="${DB_PASSWORD:-}"     # OK if blank when using peer/.pgpass auth
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-validatordb}"

PGHOST="${DB_HOST}"
PGPORT="${DB_PORT}"
PGUSER="${DB_USER}"
PGDATABASE="${DB_NAME}"

# Export for dropdb/createdb/psql
export PGPASSWORD="${DB_PASSWORD}"

# --- Find repo root and move there ---
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

# --- Drop & create database ---
echo "Dropping database '${DB_NAME}' (if it exists)…"
dropdb --if-exists -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" "${DB_NAME}"

echo "Creating database '${DB_NAME}' owned by '${DB_USER}'…"
createdb -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -O "${DB_USER}" "${DB_NAME}"

# --- Run migrations in backend ---
cd src/validator

# Alembic usually reads from env; set it explicitly to be safe.
export DATABASE_URL="postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

echo "Running Alembic migrations with uv…"
uv run alembic upgrade head

# --- Install pgq extension ---
echo "Installing pgq extension…"
# apply the output of pgq install --dry-run
pgq install --dry-run | psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}"

echo "✅ Done."

