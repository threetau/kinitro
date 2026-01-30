# ProcTHOR Evaluation Environment for Affinetes
#
# This container runs AI2-THOR procedural house simulations and queries
# miner policy endpoints to get actions. Used by validators via affinetes.
#
# Supported environments:
#   - procthor/v0 (with task types: PICKUP, PLACE, OPEN, CLOSE, TOGGLE_ON, TOGGLE_OFF)
#
# IMPORTANT: Platform Requirements:
#   AI2-THOR's Unity binary only works on native x86_64 Linux. It will NOT work:
#   - On ARM64 hosts (Apple Silicon Macs)
#   - Under QEMU x86_64 emulation (Unity's Mono JIT crashes)
#   - Without a display server (Xvfb) or GPU (CloudRendering)
#
#   For ProcTHOR evaluations, deploy to native x86_64 Linux with either:
#   - Xvfb for software rendering (slower but works without GPU)
#   - nvidia-docker with CloudRendering (faster, requires NVIDIA GPU)
#
# Build with the kinitro CLI:
#   kinitro env build procthor --tag kinitro/procthor:v1
#
# The CLI command copies the required kinitro/environments module into
# this directory before building, then cleans up afterward.
#
# The container will be managed by affinetes, which injects its HTTP
# server to expose the Actor class methods.

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for AI2-THOR (X11/Xvfb for headless rendering)
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
    # For headless rendering
    mesa-utils \
    xvfb \
    x11-utils \
    libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# AI2-THOR configuration for headless mode
ENV DISPLAY=:0
# Note: AI2THOR_PLATFORM is NOT set here - the environment.py code will:
# 1. Auto-detect Vulkan/GPU and use CloudRendering if available (faster)
# 2. Fall back to Linux64 + Xvfb if CloudRendering fails
# Set AI2THOR_PLATFORM=Linux64 at runtime to skip CloudRendering attempt.

# Force software rendering when no GPU available
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MESA_GL_VERSION_OVERRIDE=3.3

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pre-download AI2-THOR Linux64 binaries (~800MB each) to avoid runtime download.
# NOTE: These binaries only work on native x86_64 Linux, not under emulation.
# We download multiple commits to support different use cases:
# 1. Default ai2thor commit (for standard iTHOR scenes)
# 2. ProcTHOR nanna branch commit (for procedural house generation)
#    - fdb56f1c53c9 is the most recent nanna commit with a build available
# At runtime, when branch="nanna" is specified, AI2-THOR will query GitHub
# for recent commits and select the first one with an available build.
RUN python -c "\
import os; \
import ai2thor.build; \
import ai2thor.platform; \
releases_dir = os.path.join(os.path.expanduser('~'), '.ai2thor/releases'); \
# Download default ai2thor binary \
build = ai2thor.build.Build(ai2thor.platform.Linux64, ai2thor.build.COMMIT_ID, False, releases_dir=releases_dir); \
build.download(); \
print(f'Downloaded default ai2thor: {ai2thor.build.COMMIT_ID}'); \
# Download ProcTHOR nanna branch commit (most recent with build available) \
# This commit is required for procedural house generation (branch='nanna', scene='Procedural') \
nanna_commit = 'fdb56f1c53c9a4a9d7bc1babcf81cdb6c232890a'; \
build2 = ai2thor.build.Build(ai2thor.platform.Linux64, nanna_commit, False, releases_dir=releases_dir); \
build2.download(); \
print(f'Downloaded nanna commit: {nanna_commit}'); \
"

# Copy the kinitro environments package (copied by env build command)
# This is a subset of the main kinitro package - only environments/ is needed
COPY kinitro /app/kinitro

# Set PYTHONPATH so kinitro package is importable
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Copy the eval environment Actor
COPY env.py /app/

# Affinetes will inject its HTTP server - do not set CMD/ENTRYPOINT
