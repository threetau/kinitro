# MetaWorld Evaluation Environment for Affinetes
#
# This container runs MuJoCo-based manipulation simulations and queries
# miner policy endpoints to get actions. Used by validators via affinetes.
#
# Supported environments:
#   - metaworld/reach-v3
#   - metaworld/push-v3
#   - metaworld/pick-place-v3
#   - metaworld/door-open-v3
#   - metaworld/drawer-open-v3
#   - metaworld/drawer-close-v3
#   - metaworld/button-press-v3
#   - metaworld/peg-insert-v3
#
# Platform: Works on any platform (Linux, macOS, Windows)
#
# Build with the kinitro CLI:
#   kinitro build-env metaworld --tag kinitro/metaworld:v1
#
# The CLI command copies the required kinitro/environments module into
# this directory before building, then cleans up afterward.
#
# The container will be managed by affinetes, which injects its HTTP
# server to expose the Actor class methods.

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for MuJoCo rendering (EGL/OpenGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    # OpenGL/Mesa libraries for MuJoCo rendering
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

# Set MuJoCo rendering to use EGL (for headless GPU rendering)
ENV MUJOCO_GL=egl

# Force software rendering when no GPU available
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MESA_GL_VERSION_OVERRIDE=3.3

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the kinitro environments package (copied by build-env command)
# This is a subset of the main kinitro package - only environments/ is needed
COPY kinitro /app/kinitro

# Set PYTHONPATH so kinitro package is importable
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Copy the eval environment Actor
COPY env.py /app/

# Affinetes will inject its HTTP server - do not set CMD/ENTRYPOINT
