# Kinitro Evaluation Environment for Affinetes
#
# This container runs robotics simulations and queries miner policy
# endpoints to get actions. Used by validators via affinetes.
#
# Supported environments:
#   - MetaWorld (MuJoCo-based manipulation tasks) - works on all platforms
#   - ProcTHOR (AI2-THOR procedural house environments) - requires native x86_64 Linux
#
# IMPORTANT: ProcTHOR/AI2-THOR Requirements:
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
# - AI2-THOR (X11/Xvfb for headless rendering)
# - Git for pip installs
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

# Set MuJoCo rendering to use EGL (for headless GPU rendering)
ENV MUJOCO_GL=egl

# AI2-THOR configuration for headless mode
ENV DISPLAY=:0
# Skip CloudRendering attempt (requires Vulkan/GPU)
ENV AI2THOR_PLATFORM=Linux64
# Force software rendering when no GPU available
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MESA_GL_VERSION_OVERRIDE=3.3

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pre-download AI2-THOR Linux64 binary (~800MB) to avoid runtime download.
# NOTE: This binary only works on native x86_64 Linux, not under emulation.
RUN python -c "\
import os; \
import ai2thor.build; \
import ai2thor.platform; \
releases_dir = os.path.join(os.path.expanduser('~'), '.ai2thor/releases'); \
build = ai2thor.build.Build(ai2thor.platform.Linux64, ai2thor.build.COMMIT_ID, False, releases_dir=releases_dir); \
build.download(); \
"

# Copy the kinitro environments package (copied by build-eval-env command)
# This is a subset of the main kinitro package - only environments/ is needed
COPY kinitro /app/kinitro

# Set PYTHONPATH so kinitro package is importable
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Copy the eval environment Actor
COPY env.py /app/

# Affinetes will inject its HTTP server - do not set CMD/ENTRYPOINT
