"""Configuration management for robo-subnet."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class NetworkConfig(BaseSettings):
    """Bittensor network configuration."""

    model_config = SettingsConfigDict(env_prefix="ROBO_")

    network: str = Field(default="finney", description="Network: finney, test, or local")
    netuid: int = Field(default=1, description="Subnet UID")
    wallet_name: str = Field(default="default", description="Wallet name")
    hotkey_name: str = Field(default="default", description="Hotkey name")


class ValidatorConfig(BaseSettings):
    """Validator-specific configuration."""

    model_config = SettingsConfigDict(env_prefix="ROBO_")

    # Network settings (inherited concept)
    network: str = Field(default="finney")
    netuid: int = Field(default=1)
    wallet_name: str = Field(default="default")
    hotkey_name: str = Field(default="default")

    # Evaluation settings
    episodes_per_env: int = Field(
        default=50, description="Number of episodes per environment per evaluation cycle"
    )
    max_timesteps_per_episode: int = Field(default=500, description="Maximum timesteps per episode")
    action_timeout_ms: int = Field(
        default=50, description="Timeout for miner action responses in milliseconds"
    )
    eval_interval_seconds: int = Field(
        default=3600, description="Seconds between evaluation cycles"
    )
    max_concurrent_episodes: int = Field(
        default=10, description="Maximum concurrent episodes per miner"
    )

    # Scoring settings
    pareto_temperature: float = Field(
        default=1.0, description="Softmax temperature for weight conversion"
    )

    # Infrastructure
    basilica_api_token: str | None = Field(
        default=None, description="Basilica API token for container execution"
    )
    chutes_api_key: str | None = Field(default=None, description="Chutes API key for LLM inference")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")


class MinerConfig(BaseSettings):
    """Miner-specific configuration."""

    model_config = SettingsConfigDict(env_prefix="ROBO_")

    network: str = Field(default="finney")
    netuid: int = Field(default=1)
    wallet_name: str = Field(default="default")
    hotkey_name: str = Field(default="default")

    # Model settings
    huggingface_repo: str | None = Field(default=None, description="HuggingFace model repo")
    model_revision: str | None = Field(default=None, description="Model revision/commit SHA")
    chute_id: str | None = Field(default=None, description="Chutes deployment ID")

    # Docker settings
    docker_registry: str = Field(default="docker.io", description="Docker registry")
    docker_username: str | None = Field(default=None, description="Docker Hub username")
