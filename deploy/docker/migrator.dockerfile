# syntax=docker/dockerfile:1.7
#
# Lightweight image that only runs the validator database migrations. Useful as
# a Kubernetes Job executed during rollouts or auto-update hooks.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
        git \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "uv>=0.4.18"

COPY pyproject.toml README.md uv.lock /app/
RUN uv export --format requirements.txt --locked --no-dev --no-hashes \
        --no-emit-project --no-emit-workspace --no-emit-local \
        --output-file requirements.lock \
    && uv pip install --no-cache-dir -r requirements.lock

COPY scripts /app/scripts
COPY src /app/src

ENV PYTHONPATH=/app/src

ENTRYPOINT ["/app/scripts/migrate_validator_db.sh"]
