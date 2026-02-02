"""Configuration for the Scheduler service."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SchedulerConfig(BaseSettings):
    """Scheduler service configuration."""

    model_config = SettingsConfigDict(env_prefix="KINITRO_SCHEDULER_")

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/kinitro",
        description="PostgreSQL connection URL",
    )

    # Bittensor network (for reading miner commitments)
    network: str = Field(default="finney", description="Bittensor network")
    netuid: int = Field(default=1, description="Subnet UID")

    # Evaluation cycle settings
    eval_interval_seconds: int = Field(
        default=3600,
        description="Seconds between evaluation cycles",
    )
    episodes_per_env: int = Field(
        default=50,
        description="Number of episodes per environment per cycle",
    )

    # Scoring settings
    pareto_temperature: float = Field(
        default=1.0,
        description="Softmax temperature for weight conversion",
    )

    # First-commit advantage settings
    threshold_z_score: float = Field(
        default=1.5,
        description="Z-score for threshold calculation (~87% confidence)",
    )
    threshold_min_gap: float = Field(
        default=0.02,
        description="Minimum improvement required to beat earlier miner (2%)",
    )
    threshold_max_gap: float = Field(
        default=0.10,
        description="Maximum improvement cap (10%)",
    )

    # Task pool settings
    task_stale_threshold_seconds: int = Field(
        default=300,
        description="Time after which assigned tasks are considered stale",
    )
    cycle_timeout_seconds: int = Field(
        default=7200,
        description="Maximum time to wait for a cycle to complete",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
