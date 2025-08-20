"""
Configuration management for the Storb RL evaluator orchestrator.

This module uses Dynaconf to load configuration from TOML (or other
supported formats). Call `load_config(path)` at startup to load a
configuration file; use `get_config()` anywhere to get the
`OrchestratorConfig` dataclass instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from dynaconf import Dynaconf


@dataclass
class StorbRLConfig:
    database_url: str = "postgresql://localhost/storb_eval"
    ray_address: Optional[str] = None
    max_workers: int = 10
    worker_timeout_seconds: int = 1800
    worker_cpu_resources: float = 2.0
    worker_memory_gb: float = 4.0

    queue_poll_interval: float = 1.0
    max_retry_attempts: int = 3
    job_timeout_seconds: int = 3600

    default_episodes: int = 100
    default_max_steps: int = 200
    default_num_workers: int = 1

    submissions_cache_dir: Path = Path("/tmp/storb_submissions")

    kubernetes_namespace: str = "default"
    agent_base_image: str = "ghcr.io/storb-tech/agent-runner"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StorbRLConfig":
        # Only pick known keys; ignore others
        kwargs: Dict[str, Any] = {}
        for field in cls.__dataclass_fields__:
            if field in data:
                kwargs[field] = data[field]
        # Coerce Path field
        if "submissions_cache_dir" in kwargs:
            kwargs["submissions_cache_dir"] = Path(kwargs["submissions_cache_dir"])
        return cls(**kwargs)

    def validate(self) -> None:
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if self.worker_timeout_seconds <= 0:
            raise ValueError("worker_timeout_seconds must be positive")
        if self.default_episodes <= 0:
            raise ValueError("default_episodes must be positive")
        if self.default_max_steps <= 0:
            raise ValueError("default_max_steps must be positive")
        try:
            self.submissions_cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


# Module-level singleton
_GLOBAL_CONFIG: Optional[StorbRLConfig] = None


def load_config(path: Optional[str | Path] = None) -> StorbRLConfig:
    """Load configuration using Dynaconf from the given file path.

    If `path` is None the loader will try a few default locations:
    - <repo-root>/config.toml
    - ./config.toml
    """
    global _GLOBAL_CONFIG

    candidates = []
    if path:
        candidates.append(Path(path))
    else:
        repo_level = Path(__file__).resolve().parents[2] / "config.toml"
        candidates.append(repo_level)
        candidates.append(Path.cwd() / "config.toml")

    files = [str(p) for p in candidates if p.exists()]

    if files:
        settings = Dynaconf(settings_files=files, environments=True, load_dotenv=False)
        raw = settings.as_dict()  # flatten
        # Dynaconf may nest under environments; prefer top-level 'storb_rl' table
        data = raw.get("storb_rl") or raw
        cfg = StorbRLConfig.from_dict(data)
        cfg.validate()
        _GLOBAL_CONFIG = cfg
        return cfg

    # No file found; use defaults
    cfg = StorbRLConfig()
    cfg.validate()
    _GLOBAL_CONFIG = cfg
    return cfg


def get_config() -> StorbRLConfig:
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        return load_config()
    return _GLOBAL_CONFIG
