# Robotics Evaluation Environment for Affinetes
#
# This container runs MuJoCo/MetaWorld simulation and queries miner
# policy endpoints to get actions. Used by validators via affinetes.
#
# Build with affinetes:
#   import affinetes as af
#   af.build_image_from_env('eval-env', 'robo-subnet/eval-env:v1')
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

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the robo environments package (subset needed for eval)
COPY robo /app/robo

# Set PYTHONPATH so robo package is importable
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Copy the eval environment Actor
COPY env.py /app/

# Affinetes will inject its HTTP server - do not set CMD/ENTRYPOINT
