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
    eval_image: str = Field(
        default="kinitro/eval-env:v1",
        description="Docker image for evaluation environment",
    )
    eval_mode: str = Field(
        default="docker",
        description="Evaluation mode: 'docker' or 'basilica'",
    )
    eval_mem_limit: str = Field(
        default="8g",
        description="Memory limit for evaluation container",
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
