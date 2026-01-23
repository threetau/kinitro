# Dockerfile for Kinitro Validator
#
# Build: docker build -t kinitro:latest .

FROM python:3.11-slim

# Install system dependencies for MuJoCo and robotics simulation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Set MuJoCo environment variables
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy source code
COPY kinitro/ ./kinitro/

# Default command
CMD ["kinitro", "--help"]
