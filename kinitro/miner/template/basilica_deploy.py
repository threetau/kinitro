"""
Basilica Deployment for Robotics Policy Server

This script deploys your robotics policy to Basilica's GPU serverless platform.
The policy is downloaded from HuggingFace and served via FastAPI.

DEPLOYMENT OPTIONS:

1. Using this script directly:
    BASILICA_API_TOKEN=... HF_REPO=user/policy HF_REVISION=abc123 python basilica_deploy.py

2. Using kinitro CLI (recommended):
    kinitro basilica-push --repo user/policy --revision abc123

After deployment, commit your policy on-chain:
    kinitro commit --endpoint YOUR_DEPLOYMENT_URL --netuid YOUR_NETUID

Environment Variables:
    BASILICA_API_TOKEN: Your Basilica API token (required)
    HF_REPO: HuggingFace repository ID (e.g., "user/my-policy")
    HF_REVISION: HuggingFace commit SHA
    DEPLOYMENT_NAME: Custom deployment name (optional, derived from repo)
    HF_TOKEN: HuggingFace token for private repos (optional)
"""

import os
import sys


def deploy():
    """Deploy the policy server to Basilica."""
    try:
        import basilica
        from basilica import BasilicaClient
    except ImportError:
        print("Error: basilica-sdk not installed. Run: pip install basilica-sdk")
        sys.exit(1)

    # Configuration from environment
    hf_repo = os.environ.get("HF_REPO", "")
    hf_revision = os.environ.get("HF_REVISION", "main")
    hf_token = os.environ.get("HF_TOKEN", "")

    if not hf_repo:
        print("Error: HF_REPO environment variable is required")
        sys.exit(1)

    # Derive deployment name from repo (e.g., "user/my-policy" -> "user-my-policy")
    default_name = hf_repo.replace("/", "-").lower()
    deployment_name = os.environ.get("DEPLOYMENT_NAME", default_name)

    print(f"Deploying {hf_repo}@{hf_revision[:12]}... as '{deployment_name}'")

    # Create client
    client = BasilicaClient()

    # Create cache volume to avoid re-downloading model on restarts
    cache_volume_name = f"kinitro-{deployment_name}-cache"[:63]  # Max 63 chars
    try:
        model_cache = basilica.Volume.from_name(cache_volume_name, create_if_missing=True)
        print(f"Using cache volume: {cache_volume_name}")
    except Exception as e:
        print(f"Warning: Could not create cache volume: {e}")
        model_cache = None

    # Generate deployment source code
    # This code runs inside the Basilica container at startup
    hf_token_env = (
        f'os.environ.get("HF_TOKEN", "{hf_token}")' if hf_token else 'os.environ.get("HF_TOKEN")'
    )

    source_code = f'''
import os
import sys
import subprocess

print("Starting Kinitro Policy Server...")
print(f"HF_REPO: {{os.environ.get('HF_REPO', 'not set')}}")
print(f"HF_REVISION: {{os.environ.get('HF_REVISION', 'not set')}}")

# Download model from HuggingFace
from huggingface_hub import snapshot_download

hf_token = {hf_token_env}
print("Downloading model from HuggingFace...")
snapshot_download(
    "{hf_repo}",
    revision="{hf_revision}",
    local_dir="/app",
    token=hf_token,
)
print("Model downloaded successfully!")

# Add /app to Python path so we can import the policy
sys.path.insert(0, "/app")

# Start the FastAPI server
print("Starting uvicorn server on port 8000...")
subprocess.run([
    sys.executable, "-m", "uvicorn",
    "server:app",
    "--host", "0.0.0.0",
    "--port", "8000",
])
'''

    # Build deployment configuration
    deploy_kwargs = {
        "name": deployment_name,
        "source": source_code,
        "image": "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
        "port": 8000,
        "gpu_count": 1,
        "min_gpu_memory_gb": 16,
        "memory": "16Gi",
        "cpu": "4000m",  # 4 CPU cores
        "pip_packages": [
            "fastapi",
            "uvicorn",
            "numpy",
            "huggingface-hub",
            "pydantic",
            "pillow",
        ],
        "env": {
            "HF_REPO": hf_repo,
            "HF_REVISION": hf_revision,
        },
        "timeout": 600,  # 10 minute timeout for GPU scheduling
    }

    # Add HF token if provided
    if hf_token:
        deploy_kwargs["env"]["HF_TOKEN"] = hf_token

    # Add cache volume if created successfully
    if model_cache:
        deploy_kwargs["volumes"] = {"/root/.cache/huggingface": model_cache}

    # Deploy
    print("\nDeploying to Basilica...")
    try:
        deployment = client.deploy(**deploy_kwargs)
    except Exception as e:
        print(f"Deployment failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("DEPLOYMENT SUCCESSFUL")
    print("=" * 60)
    print(f"  Name: {deployment.name}")
    print(f"  URL: {deployment.url}")
    print(f"  State: {deployment.state}")
    print("=" * 60)
    print("\nNext step - commit on-chain:")
    print(f"  kinitro commit --repo {hf_repo} --revision {hf_revision} \\")
    print(f"    --endpoint {deployment.url} --netuid YOUR_NETUID")
    print()

    return deployment


if __name__ == "__main__":
    deploy()
