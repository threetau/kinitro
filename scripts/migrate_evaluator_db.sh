#!/usr/bin/env bash
set -euo pipefail

# --- Config (from environment, with sane defaults) ---
DB_USER="${DB_USER:-myuser}"
DB_PASSWORD="${DB_PASSWORD:-}"     # OK if blank when using peer/.pgpass auth
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-evaluatordb}"

PGHOST="${DB_HOST}"
PGPORT="${DB_PORT}"
PGUSER="${DB_USER}"
PGDATABASE="${DB_NAME}"

# Export for psql/pgq commands
export PGPASSWORD="${DB_PASSWORD}"

# --- Find repo root and move there ---
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

# --- Check if database exists ---
echo "Checking if database '${DB_NAME}' exists..."
if ! psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -lqt | cut -d \| -f 1 | grep -qw "${DB_NAME}"; then
    echo "❌ Database '${DB_NAME}' does not exist. Please create it first."
    exit 1
fi

# --- Run migrations in evaluator ---
cd src/evaluator

# Alembic usually reads from env; set it explicitly to be safe.
export DATABASE_URL="postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

echo "Running Alembic migrations with uv…"
uv run alembic upgrade head

echo "✅ Evaluator database migrations completed successfully."
