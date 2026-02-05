# Affinetes Evaluation Environment Template
#
# Build with the kinitro CLI:
#   kinitro env build myenv --tag kinitro/myenv:v1
#
# The CLI copies kinitro/environments into this directory before building.

FROM python:3.11-slim

WORKDIR /app

# ============================================================================
# System dependencies
# ============================================================================
# TODO: Add system packages needed by your simulator
# Common packages for rendering:
#   libgl1, libegl1, libglib2.0-0 (OpenGL/EGL)
#   libsm6, libxext6, libxrender1 (X11)
# ============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Python dependencies
# ============================================================================
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# ============================================================================
# Kinitro package (copied by `kinitro env build`)
# ============================================================================
COPY kinitro /app/kinitro

ENV PYTHONPATH="/app:${PYTHONPATH}"

# ============================================================================
# Actor class
# ============================================================================
COPY env.py /app/

# Affinetes injects its HTTP server - do not set CMD/ENTRYPOINT
