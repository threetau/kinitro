# syntax=docker/dockerfile:1.7
#
# Evaluator image (CUDA variant). Requires the NVIDIA Container Toolkit on
# the host or Kubernetes cluster nodes.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    MUJOCO_GL=egl \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        curl \
        git \
        libpq-dev \
        postgresql-client \
        libegl1 \
        libglew-dev \
        libgl1 \
        libosmesa6 \
        libosmesa6-dev \
        patchelf \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
    && python3.12 -m ensurepip --upgrade \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.12 /usr/bin/python && \
    ln -s /usr/local/bin/pip3.12 /usr/bin/pip

RUN pip install --no-cache-dir "uv>=0.4.18"

COPY pyproject.toml README.md uv.lock /app/
RUN uv export --format requirements.txt --locked --no-dev --no-hashes \
        --no-emit-project --no-emit-workspace --no-emit-local \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
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
