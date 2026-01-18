# Dockerfile for Robotics Subnet Miner Policy Server
#
# This creates a FastAPI server that exposes your policy for validator queries.
#
# Build: docker build -t your-username/robo-policy:v1 .
# Test:  docker run -p 8000:8000 your-username/robo-policy:v1
# Push:  docker push your-username/robo-policy:v1
#
# Deploy to Chutes: see chute.py for configuration

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

# Copy policy server code
COPY server.py .
COPY policy.py .

# Copy model weights (uncomment and modify as needed)
# COPY model.pt .
# COPY models/ ./models/

# Expose port for FastAPI
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
