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

    # Endpoint decryption (for encrypted miner commitments)
    backend_private_key: str | None = Field(
        default=None,
        description="X25519 private key (hex) for decrypting miner endpoints. "
        "Required if miners use encrypted commitments.",
    )
    backend_private_key_file: str | None = Field(
        default=None,
        description="Path to file containing the backend private key (hex).",
    )

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
        default=900,
        description="Time after which assigned tasks are considered stale (15 min for Basilica)",
    )
    cycle_timeout_seconds: int = Field(
        default=7200,
        description="Maximum time to wait for a cycle to complete",
    )

    # Cycle isolation
    cleanup_incomplete_cycles: bool = Field(
        default=True,
        description="Cancel incomplete cycles and their tasks on startup. "
        "This ensures cycle isolation when the scheduler restarts.",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
