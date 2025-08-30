# Dockerfile for the submission containers
FROM python:3.12-slim

# Install system dependencies (if torch etc. need BLAS, gcc, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl \
    libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

# Install ML dependencies (pin versions for reproducibility)
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    transformers==4.44.0 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    pycapnp==2.0.0 \
    metaworld==3.0.0 \
    gymnasium==1.2.0 \
    Pillow==10.4.0

# Create workspace directories
RUN mkdir -p /workspace/submission

# Copy in the driver files
WORKDIR /workspace
COPY ./submission_template/main.py /templates/main.py
COPY ./src/evaluator/rpc/agent.capnp /templates/agent.capnp
COPY ./src/evaluator/agent_interface.py /templates/agent_interface.py
COPY ./src/evaluator/rpc/server.py /templates/agent_server.py
# No additional JSON-RPC files needed for Cap'n Proto
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000"]
