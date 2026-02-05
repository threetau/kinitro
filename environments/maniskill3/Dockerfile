# ManiSkill3 Evaluation Environment for Affinetes
#
# This container runs SAPIEN-based manipulation simulations and queries
# miner policy endpoints to get actions. Used by validators via affinetes.
#
# Supported environments:
#   - maniskill3/pick-cube-v1
#   - maniskill3/stack-cube-v1
#   - maniskill3/peg-insertion-v1
#   - maniskill3/push-cube-v1
#   - maniskill3/pull-cube-v1
#   - maniskill3/lift-cube-v1
#   - maniskill3/turn-faucet-v1
#   - maniskill3/open-cabinet-door-v1
#
# Platform: Linux (x86_64), requires Vulkan for GPU rendering
#
# Build with the kinitro CLI:
#   kinitro env build maniskill3 --tag kinitro/maniskill3:v1
#
# The CLI command copies the required kinitro/environments module into
# this directory before building, then cleans up afterward.
#
# The container will be managed by affinetes, which injects its HTTP
# server to expose the Actor class methods.

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for SAPIEN/Vulkan rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    # OpenGL/Mesa libraries
    libgl1 \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libegl1 \
    libegl-mesa0 \
    libglvnd0 \
    libglx0 \
    libglx-mesa0 \
    mesa-utils \
    # Vulkan support for SAPIEN
    libvulkan1 \
    libvulkan-dev \
    vulkan-tools \
    mesa-vulkan-drivers \
    # X11 for headless rendering fallback
    xvfb \
    x11-utils \
    && rm -rf /var/lib/apt/lists/*

# Set rendering environment variables
# SAPIEN can use Vulkan (preferred) or EGL
ENV DISPLAY=:0
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MESA_GL_VERSION_OVERRIDE=3.3

# For headless Vulkan rendering
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the kinitro environments package (copied by env build command)
# This is a subset of the main kinitro package - only environments/ is needed
COPY kinitro /app/kinitro

# Set PYTHONPATH so kinitro package is importable
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Copy the eval environment Actor
COPY env.py /app/

# Affinetes will inject its HTTP server - do not set CMD/ENTRYPOINT
