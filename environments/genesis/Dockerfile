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
    # OSMesa for CPU-only software rendering fallback
    libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

# Render platform is auto-detected at runtime (EGL for GPU, OSMesa for CPU).
# Do NOT hardcode PYOPENGL_PLATFORM here — see base.py:_detect_render_platform().
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV MESA_GL_VERSION_OVERRIDE=3.3

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Clone MuJoCo Menagerie (sparse checkout -- only robot dirs needed)
# This provides MJCF assets for supported robots (G1, Go2, Franka, etc.)
# Pinned to a known-good commit for reproducible builds.
ARG MENAGERIE_COMMIT=a03e87bf13502b0b48ebbf2808928fd96ebf9cf3
RUN git clone --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git /opt/menagerie && \
    cd /opt/menagerie && \
    git checkout ${MENAGERIE_COMMIT} && \
    git sparse-checkout set unitree_g1 && \
    echo "Menagerie assets installed"

ENV GENESIS_MENAGERIE_PATH=/opt/menagerie

# Run as non-root user
RUN groupadd --gid 1000 genesis && \
    useradd --uid 1000 --gid genesis --create-home genesis

# Pre-warm Taichi kernel cache for Genesis physics.
# Runs a minimal headless scene build to JIT-compile all physics kernels
# and bake them into the image layer (~36s savings per container startup).
# Placed before COPY kinitro/env.py so app code changes don't invalidate
# this slow layer.
COPY warmup_kernel_cache.py /tmp/warmup_kernel_cache.py
RUN chown genesis:genesis /tmp/warmup_kernel_cache.py
USER genesis
RUN python /tmp/warmup_kernel_cache.py
USER root
RUN rm /tmp/warmup_kernel_cache.py

# Copy the kinitro environments package (copied by env build command)
# This is a subset of the main kinitro package - only environments/ is needed
COPY kinitro /app/kinitro

# Set PYTHONPATH so kinitro package is importable
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Copy the eval environment Actor
COPY env.py /app/

RUN chown -R genesis:genesis /app
USER genesis

# Affinetes will inject its HTTP server - do not set CMD/ENTRYPOINT
