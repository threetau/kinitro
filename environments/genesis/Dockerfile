# Genesis Evaluation Environment for Affinetes
#
# This container runs Genesis physics simulations and queries miner policy
# endpoints to get actions. Used by validators via affinetes.
#
# Supported environments:
#   - genesis/g1-v0 (Unitree G1 humanoid: navigate, pickup, place, push)
#
# Single Docker image serves all genesis/* env_ids. The env_id selects
# which RobotConfig + subclass to instantiate at runtime.
#
# Build with the kinitro CLI:
#   kinitro env build genesis --tag kinitro/genesis:v1
#
# The CLI command copies the required kinitro/environments module into
# this directory before building, then cleans up afterward.
#
# The container will be managed by affinetes, which injects its HTTP
# server to expose the Actor class methods.

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Genesis rendering (EGL/OpenGL headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    # OpenGL/Mesa libraries for headless rendering
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
    && rm -rf /var/lib/apt/lists/*

# EGL for headless rendering
ENV PYOPENGL_PLATFORM=egl
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Force software rendering when no GPU available
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MESA_GL_VERSION_OVERRIDE=3.3

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Clone MuJoCo Menagerie (sparse checkout -- only robot dirs needed)
# This provides MJCF assets for supported robots (G1, Go2, Franka, etc.)
RUN git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git /opt/menagerie && \
    cd /opt/menagerie && \
    git sparse-checkout set unitree_g1 && \
    echo "Menagerie assets installed"

ENV GENESIS_MENAGERIE_PATH=/opt/menagerie

# Copy the kinitro environments package (copied by env build command)
# This is a subset of the main kinitro package - only environments/ is needed
COPY kinitro /app/kinitro

# Set PYTHONPATH so kinitro package is importable
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Copy the eval environment Actor
COPY env.py /app/

# Affinetes will inject its HTTP server - do not set CMD/ENTRYPOINT
