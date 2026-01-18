# Dockerfile for Robotics Subnet Miner Policy
#
# Build: docker build -t your-username/robo-policy:v1 .
# Push:  docker push your-username/robo-policy:v1

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy policy code and weights
COPY env.py .
# COPY policy.pt .        # Uncomment if you have saved model weights
# COPY your_model/ ./     # Or copy a model directory

# The policy is loaded when the container starts
# Validators will call the RobotActor class methods
