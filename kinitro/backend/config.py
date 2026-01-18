"""Configuration for the backend service."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendConfig(BaseSettings):
    """Backend service configuration."""

    model_config = SettingsConfigDict(env_prefix="KINITRO_BACKEND_")

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/kinitro",
        description="PostgreSQL connection URL",
    )

    # API Server
    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, description="API server port")

    # Bittensor network (for reading miner commitments)
    network: str = Field(default="finney", description="Bittensor network")
    netuid: int = Field(default=1, description="Subnet UID")

    # Evaluation settings
    eval_interval_seconds: int = Field(
        default=3600,
        description="Seconds between evaluation cycles",
    )
    episodes_per_env: int = Field(
        default=50,
        description="Number of episodes per environment per cycle",
    )
    max_timesteps_per_episode: int = Field(
        default=500,
        description="Maximum timesteps per episode",
    )
    action_timeout_ms: int = Field(
        default=100,
        description="Timeout for miner action responses",
    )

    # Scoring settings
    pareto_temperature: float = Field(
        default=1.0,
        description="Softmax temperature for weight conversion",
    )

    # Infrastructure
    basilica_api_token: str | None = Field(
        default=None,
        description="Basilica API token for container execution",
    )

    # Affinetes settings
    eval_mode: str = Field(
        default="docker",
        description="Evaluation mode: 'docker' or 'basilica'",
    )
    eval_image: str = Field(
        default="kinitro/eval-env:v1",
        description="Docker image for evaluation environment",
    )
    eval_mem_limit: str = Field(
        default="8g",
        description="Memory limit for evaluation container",
    )
    eval_hosts: list[str] = Field(
        default_factory=lambda: ["localhost"],
        description="Docker hosts for evaluation (can include SSH remotes)",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
