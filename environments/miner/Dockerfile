# Miner Policy Runner for Affinetes
#
# This container downloads miner policies from HuggingFace and runs them
# as evaluation endpoints. Used by executors to create miner deployments.
#
# Environment Variables (set at runtime):
#   HF_REPO     - HuggingFace repository (e.g., "user/policy")
#   HF_REVISION - HuggingFace commit SHA
#   HF_TOKEN    - Optional HuggingFace token for private repos
#
# Platform: Works on any platform (Linux, macOS, Windows)
#
# Build with the kinitro CLI:
#   kinitro env build miner --tag kinitro/miner-runner:v1
#
# The container will be managed by affinetes, which injects its HTTP
# server to expose the Actor class methods.

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the Actor implementation
COPY env.py /app/

# Set PYTHONPATH
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Affinetes will inject its HTTP server - do not set CMD/ENTRYPOINT
