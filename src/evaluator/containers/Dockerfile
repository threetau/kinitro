# Dockerfile for the submission containers
FROM python:3.12-slim

# Install system dependencies (if torch etc. need BLAS, gcc, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl iptables \
    libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

# Install minimal runtime deps for the RPC layer only.
# Miner-specified packages are installed at runtime into a per-submission venv.
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pycapnp==2.0.0

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
