# Kinitro Evaluation Environment for Affinetes
#
# This container runs robotics simulations and queries miner policy
# endpoints to get actions. Used by validators via affinetes.
#
# Supported environments:
#   - MetaWorld (MuJoCo-based manipulation tasks)
#   - ProcTHOR (AI2-THOR procedural house environments)
#
# Build with the kinitro CLI:
#   kinitro build-eval-env --tag kinitro/eval-env:v1
#
# The CLI command copies the required kinitro/environments module into
# this directory before building, then cleans up afterward.
#
# The container will be managed by affinetes, which injects its HTTP
# server to expose the Actor class methods.

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for:
# - MuJoCo rendering (EGL/OpenGL)
# - AI2-THOR (X11/Vulkan)
# - Git for pip installs
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
    # AI2-THOR dependencies
    xvfb \
    libvulkan1 \
    vulkan-tools \
    mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

# Set MuJoCo rendering to use EGL (for headless GPU rendering)
ENV MUJOCO_GL=egl

# AI2-THOR headless mode via Xvfb
ENV DISPLAY=:0

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the kinitro environments package (copied by build-eval-env command)
# This is a subset of the main kinitro package - only environments/ is needed
COPY kinitro /app/kinitro

# Set PYTHONPATH so kinitro package is importable
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Copy the eval environment Actor
COPY env.py /app/

# Affinetes will inject its HTTP server - do not set CMD/ENTRYPOINT
