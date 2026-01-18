# Robotics Evaluation Environment for Affinetes
#
# This container runs MuJoCo/MetaWorld simulation and queries miner
# policy endpoints to get actions. Used by validators via affinetes.
#
# Build:
#   docker build -t robo-subnet/eval-env:v1 -f eval-env/Dockerfile .
#
# The container will be managed by affinetes, which injects its HTTP
# server to expose the Actor class methods.

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for MuJoCo rendering + git for pip installs
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libegl1 \
    libglvnd0 \
    libglx0 \
    && rm -rf /var/lib/apt/lists/*

# Set MuJoCo rendering to use EGL (for headless GPU rendering)
ENV MUJOCO_GL=egl

# Copy requirements first for better caching
COPY eval-env/requirements.txt /app/eval-env/
RUN pip install --no-cache-dir -r /app/eval-env/requirements.txt

# Copy the robo package
COPY robo /app/robo
COPY pyproject.toml /app/

# Install minimal dependencies needed for robo package
RUN pip install --no-cache-dir pydantic pydantic-settings

# Set PYTHONPATH so robo package is importable
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Copy the eval environment Actor
COPY eval-env/env.py /app/eval-env/

# Set working directory to eval-env so affinetes finds env.py
WORKDIR /app/eval-env

# Affinetes will inject its HTTP server - do not set CMD/ENTRYPOINT
