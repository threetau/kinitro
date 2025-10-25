# syntax=docker/dockerfile:1.7
#
# Backend image responsible for serving the REST/WebSocket API and running
# Alembic migrations for the core backend schema.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Install system dependencies required for compiled Python packages such as
# psycopg and capnp.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libcapnp-dev \
        libpq-dev \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install uv for lock-file driven dependency resolution.
RUN pip install --no-cache-dir "uv>=0.4.18"

# Copy metadata and lock files first for better build caching.
COPY pyproject.toml README.md uv.lock /app/

# Export the lock file to requirements and install production dependencies.
RUN uv export --format requirements.txt --locked --no-dev --no-hashes \
        --no-emit-project --no-emit-workspace --no-emit-local \
        --output-file requirements.lock \
    && uv pip install --system --no-cache-dir -r requirements.lock

# Copy project sources and helper scripts used by the entrypoint.
COPY src /app/src
COPY scripts /app/scripts

ENV KINITRO_HOME=/var/lib/kinitro \
    PATH="/app/.local/bin:${PATH}" \
    PYTHONPATH=/app/src

# Create non-root user for runtime and ensure ownership of relevant paths.
RUN useradd --system --create-home --home-dir /var/lib/kinitro kinitro \
    && chown -R kinitro:kinitro /app /var/lib/kinitro

COPY deploy/docker/entrypoint-backend.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown kinitro:kinitro /entrypoint.sh

USER kinitro

# Run migrations (optional) and launch the backend API.
ENTRYPOINT ["/entrypoint.sh"]
