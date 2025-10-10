# syntax=docker/dockerfile:1.7
#
# Runtime image for miner submissions. Bundles the RPC bridge templates and
# utilities required by submission pods. The init container clones the miner's
# repository and installs the submission-specific requirements.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SUBMISSION_ROOT=/workspace/submission

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Ship commonly used RL libraries to reduce per-submission install cost.
COPY deploy/docker/requirements-miner.txt /tmp/requirements-miner.txt
RUN pip install -r /tmp/requirements-miner.txt

# Seed the submission templates so init containers can copy them over.
COPY submission_template /templates

RUN useradd --system --create-home --home-dir /workspace miner \
    && mkdir -p "${SUBMISSION_ROOT}" \
    && chown -R miner:miner /workspace /templates

USER miner
WORKDIR ${SUBMISSION_ROOT}

CMD ["python", "/workspace/submission/main.py", "--host", "0.0.0.0", "--port", "8000"]
