# syntax=docker/dockerfile:1.7
#
# Validator image: focused on running the websocket validator service and
# Alembic/pgqueuer migrations. Relies on the `uv.lock` file to produce
# reproducible installs.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# System deps needed for psycopg2 and other compiled wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
        git \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install uv for locked dependency resolution.
RUN pip install --no-cache-dir "uv>=0.4.18"

# Copy metadata and lock files first to maximize layer caching.
COPY pyproject.toml README.md uv.lock /app/

# Export lockfile to requirements.txt and install dependencies.
RUN uv export --format requirements.txt --locked --no-dev --no-hashes \
        --no-emit-project --no-emit-workspace --no-emit-local \
        --output-file requirements.lock \
    && uv pip install --no-cache-dir -r requirements.lock

# Copy the source tree (validator + shared core + scripts).
COPY src /app/src
COPY scripts /app/scripts

# The validator expects configuration/credential files to be mounted.
ENV KINITRO_HOME=/var/lib/kinitro \
    PATH="/app/.local/bin:${PATH}" \
    PYTHONPATH=/app/src

# Expose a non-root user for better isolation.
RUN useradd --system --create-home --home-dir /var/lib/kinitro kinitro \
    && chown -R kinitro:kinitro /app /var/lib/kinitro
COPY deploy/docker/entrypoint-validator.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown kinitro:kinitro /entrypoint.sh
USER kinitro

# Default command runs migrations then starts the websocket validator.

ENTRYPOINT ["/entrypoint.sh"]
