# syntax=docker/dockerfile:1.7
#
# Evaluator image (CPU variant). Bundles Ray, Gymnasium, MuJoCo, and rollout
# runtime dependencies. Use the CUDA sibling Dockerfile for GPU workloads.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    MUJOCO_GL=osmesa

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libpq-dev \
        postgresql-client \
        libgl1 \
        libglew-dev \
        libosmesa6 \
        libosmesa6-dev \
        patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "uv>=0.4.18"

COPY pyproject.toml README.md uv.lock /app/
RUN uv export --format requirements.txt --locked --no-dev --no-hashes \
        --no-emit-project --no-emit-workspace --no-emit-local \
        --output-file requirements.lock \
    && uv pip install --no-cache-dir -r requirements.lock

COPY src /app/src
COPY scripts /app/scripts

ENV KINITRO_HOME=/var/lib/kinitro \
    PATH="/app/.local/bin:${PATH}" \
    PYTHONPATH=/app/src

RUN useradd --system --create-home --home-dir /var/lib/kinitro kinitro \
    && chown -R kinitro:kinitro /app /var/lib/kinitro
COPY deploy/docker/entrypoint-evaluator.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown kinitro:kinitro /entrypoint.sh
USER kinitro

ENTRYPOINT ["/entrypoint.sh"]
