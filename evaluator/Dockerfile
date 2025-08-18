# Dockerfile for the submission containers
FROM python:3.11-slim

# Install system dependencies (if torch etc. need BLAS, gcc, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install ML dependencies (pin versions for reproducibility)
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    transformers==4.44.0 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    pycapnp==2.0.0 \
    metaworld==3.0.0 \
    gymnasium==1.2.0

# Copy in your template driver files
WORKDIR /workspace
COPY ./submission_template/ /templates/
# Default command (overridable in K8s)
CMD ["python", "main.py"]
