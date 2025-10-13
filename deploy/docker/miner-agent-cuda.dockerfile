# syntax=docker/dockerfile:1.7
#
# GPU-enabled runtime image for miner submissions.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    SUBMISSION_ROOT=/workspace/submission

WORKDIR /tmp

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3-pip \
        git \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.12 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

COPY deploy/docker/requirements-miner-cuda.txt /tmp/requirements-miner.txt
RUN pip install --no-cache-dir -r /tmp/requirements-miner.txt --extra-index-url https://download.pytorch.org/whl/cu124

COPY submission_template /templates

RUN useradd --system --create-home --home-dir /workspace miner \
    && mkdir -p "${SUBMISSION_ROOT}" \
    && chown -R miner:miner /workspace /templates

USER miner
WORKDIR ${SUBMISSION_ROOT}

CMD ["python", "/workspace/submission/main.py", "--host", "0.0.0.0", "--port", "8000"]
