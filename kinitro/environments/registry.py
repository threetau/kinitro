"""Environment registry for loading robotics environments."""

import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog

from kinitro.environments.base import RoboticsEnvironment

logger = structlog.get_logger()

# Note: Environment classes are imported lazily in factory functions to allow
# containers with partial dependencies (e.g., genesis container without metaworld)

# Path to environments directory (robo-subnet/environments/)
_ENVIRONMENTS_DIR = Path(__file__).parent.parent.parent / "environments"

# Type alias for environment factory functions
EnvFactory = Callable[..., RoboticsEnvironment]


def _make_metaworld_env(task: str) -> EnvFactory:
    """Create factory for MetaWorld environment."""

    def factory(**_kwargs: Any) -> RoboticsEnvironment:
        # Lazy import to allow containers with partial dependencies
        from kinitro.environments.metaworld_env import MetaWorldEnvironment  # noqa: PLC0415

        return MetaWorldEnvironment(task)

    return factory


def _make_genesis_env(env_cls_name: str, task: str) -> EnvFactory:
    """Create factory for Genesis environment."""

    def factory(**kwargs: Any) -> RoboticsEnvironment:
        # Lazy import to allow containers with partial dependencies
        from kinitro.environments.genesis import envs  # noqa: PLC0415

        cls = getattr(envs, env_cls_name)
        return cls(task_name=task, **kwargs)

    return factory


# ===========================================================================
# ENVIRONMENT REGISTRY
# ===========================================================================
# MetaWorld manipulation environments
# ===========================================================================

ENVIRONMENTS: dict[str, EnvFactory] = {
    # =========================================================================
    # MANIPULATION (MetaWorld V3)
    # Robot arm pick, place, push, and interact with objects
    # =========================================================================
    "metaworld/reach-v3": _make_metaworld_env("reach-v3"),
    "metaworld/push-v3": _make_metaworld_env("push-v3"),
    "metaworld/pick-place-v3": _make_metaworld_env("pick-place-v3"),
    "metaworld/door-open-v3": _make_metaworld_env("door-open-v3"),
    "metaworld/drawer-open-v3": _make_metaworld_env("drawer-open-v3"),
    "metaworld/drawer-close-v3": _make_metaworld_env("drawer-close-v3"),
    "metaworld/button-press-v3": _make_metaworld_env("button-press-topdown-v3"),
    "metaworld/peg-insert-v3": _make_metaworld_env("peg-insert-side-v3"),
    # =========================================================================
    # GENESIS (Physics simulation with humanoid, quadruped, manipulation)
    # Genesis-world engine with procedural scenes and scene-grounded tasks
    # =========================================================================
    "genesis/g1-v0": _make_genesis_env("G1Environment", "g1-v0"),
}


def get_environment(env_id: str, **kwargs: Any) -> RoboticsEnvironment:
    """
    Load a robotics environment by ID.

    Args:
        env_id: Environment identifier (e.g., 'metaworld/pick-place-v3')
        **kwargs: Extra keyword arguments passed to the environment factory.

    Returns:
        Initialized RoboticsEnvironment instance

    Raises:
        ValueError: If environment ID is not found
    """
    if env_id not in ENVIRONMENTS:
        available = list(ENVIRONMENTS.keys())
        raise ValueError(
            f"Unknown environment: {env_id}. Available environments:\n"
            + "\n".join(f"  - {e}" for e in available)
        )
    return ENVIRONMENTS[env_id](**kwargs)


def get_all_environment_ids() -> list[str]:
    """Get list of all registered environment IDs."""
    return list(ENVIRONMENTS.keys())


def get_environments_by_family(family: str) -> list[str]:
    """
    Get environment IDs for a specific family.

    Args:
        family: Environment family (currently only 'metaworld')

    Returns:
        List of environment IDs in that family
    """
    return [env_id for env_id in ENVIRONMENTS if env_id.startswith(f"{family}/")]


def _load_family_metadata() -> dict[str, dict[str, str]]:
    """Load family metadata from environments directory metadata.json files."""
    metadata = {}
    if not _ENVIRONMENTS_DIR.exists():
        return metadata

    for family_dir in _ENVIRONMENTS_DIR.iterdir():
        if not family_dir.is_dir():
            continue
        metadata_file = family_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    family_metadata = json.load(f)
                metadata[family_dir.name] = family_metadata
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    "failed_to_load_family_metadata",
                    family=family_dir.name,
                    error=str(e),
                )
    return metadata


class _FamilyMetadataCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._cache: dict[str, dict[str, str]] = {}

    def get(self) -> dict[str, dict[str, str]]:
        """Get cached family metadata, loading if necessary (thread-safe)."""
        if self._loaded:
            return self._cache
        with self._lock:
            # Double-check after acquiring lock
            if not self._loaded:
                self._cache = _load_family_metadata()
                self._loaded = True
            return self._cache


_family_metadata_cache = _FamilyMetadataCache()


def _get_family_metadata_cache() -> dict[str, dict[str, str]]:
    """Get cached family metadata, loading if necessary."""
    return _family_metadata_cache.get()


def get_available_families() -> list[str]:
    """Get list of available environment families from environments directory."""
    return list(_get_family_metadata_cache().keys())


def get_family_metadata(family: str) -> dict[str, str] | None:
    """Get display metadata for a family (name, description) from metadata.json."""
    return _get_family_metadata_cache().get(family)
