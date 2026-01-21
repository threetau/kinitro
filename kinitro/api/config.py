"""Configuration for the API service."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class APIConfig(BaseSettings):
    """API service configuration."""

    model_config = SettingsConfigDict(env_prefix="KINITRO_API_")

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/kinitro",
        description="PostgreSQL connection URL",
    )

    # API Server
    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, description="API server port")

    # Task pool settings
    task_stale_threshold_seconds: int = Field(
        default=300,
        description="Time after which assigned tasks are considered stale",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
