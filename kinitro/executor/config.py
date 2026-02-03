"""Configuration for the Executor service."""

import uuid

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def default_executor_id() -> str:
    """Generate a default executor ID."""
    return f"executor-{uuid.uuid4().hex[:8]}"


class ExecutorConfig(BaseSettings):
    """Executor service configuration."""

    model_config = SettingsConfigDict(env_prefix="KINITRO_EXECUTOR_")

    # API connection
    api_url: str = Field(
        default="http://localhost:8000",
        description="URL of the Kinitro API service",
    )

    # Executor identity
    executor_id: str = Field(
        default_factory=default_executor_id,
        description="Unique identifier for this executor",
    )

    # Task fetching
    batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of tasks to fetch at a time",
    )
    poll_interval_seconds: int = Field(
        default=5,
        description="Seconds to wait between polling for tasks",
    )
    env_ids: list[str] | None = Field(
        default=None,
        description="Filter tasks by environment IDs (None = all envs)",
    )

    # Affinetes settings for evaluation
    eval_images: dict[str, str] = Field(
        default_factory=lambda: {
            "metaworld": "kinitro/metaworld:v1",
            "procthor": "kinitro/procthor:v1",
        },
        description="Mapping of environment family to Docker image. "
        "Keys are family prefixes (e.g., 'metaworld'), values are image tags. "
        "Build images with 'kinitro env build <family>'.",
    )
    eval_mode: str = Field(
        default="docker",
        description="Evaluation mode: 'docker' or 'basilica'",
    )
    eval_mem_limit: str = Field(
        default="8Gi",
        description="Memory limit for evaluation container (use Kubernetes format: 512Mi, 8Gi)",
    )
    eval_hosts: list[str] = Field(
        default_factory=lambda: ["localhost"],
        description="Docker hosts for evaluation",
    )

    # Evaluation settings
    max_timesteps: int = Field(
        default=500,
        description="Maximum timesteps per episode",
    )
    action_timeout: float = Field(
        default=0.5,
        description="Timeout for miner action responses (seconds)",
    )
    eval_timeout: int = Field(
        default=300,
        description="Timeout for individual evaluation (seconds)",
    )
    use_images: bool = Field(
        default=True,
        description="Whether to include camera images in observations",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    # Model verification settings
    verification_enabled: bool = Field(
        default=True,
        description="Enable spot-check verification of miner models",
    )
    verification_rate: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Probability of verifying each miner (0.0 to 1.0)",
    )
    verification_tolerance: float = Field(
        default=1e-3,
        description="Relative tolerance for comparing actions",
    )
    verification_samples: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of test observations per verification",
    )
    verification_cache_dir: str | None = Field(
        default=None,
        description="Directory to cache downloaded HuggingFace models",
    )
    verification_max_repo_size_gb: float = Field(
        default=5.0,
        ge=0.1,
        le=50.0,
        description="Maximum allowed HuggingFace repo size in GB",
    )

    def get_image_for_env(self, env_id: str) -> str:
        """
        Get the Docker image for a given environment ID.

        Args:
            env_id: Environment ID (e.g., 'metaworld/pick-place-v3')

        Returns:
            Docker image tag for the environment's family

        Raises:
            ValueError: If no image is configured for the environment's family
        """
        # Extract family from env_id (e.g., 'metaworld' from 'metaworld/pick-place-v3')
        family = env_id.split("/")[0] if "/" in env_id else env_id

        if family not in self.eval_images:
            available = list(self.eval_images.keys())
            raise ValueError(
                f"No Docker image configured for environment family '{family}'. "
                f"Available families: {available}. "
                f"Set KINITRO_EXECUTOR_EVAL_IMAGES to configure."
            )

        return self.eval_images[family]
