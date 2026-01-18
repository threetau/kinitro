"""Environment registry for loading robotics environments."""

from collections.abc import Callable

import structlog

from robo.environments.base import RoboticsEnvironment

logger = structlog.get_logger()

# Type alias for environment factory functions
EnvFactory = Callable[[], RoboticsEnvironment]


def _make_metaworld_env(task: str) -> EnvFactory:
    """Create factory for MetaWorld environment."""

    def factory() -> RoboticsEnvironment:
        from robo.environments.metaworld_env import MetaWorldEnvironment

        return MetaWorldEnvironment(task)

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
}


def get_environment(env_id: str) -> RoboticsEnvironment:
    """
    Load a robotics environment by ID.

    Args:
        env_id: Environment identifier (e.g., 'metaworld/pick-place-v3')

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
    return ENVIRONMENTS[env_id]()


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


def get_available_families() -> list[str]:
    """Get list of available environment families."""
    return ["metaworld"]


def is_family_available(family: str) -> bool:
    """Check if an environment family is available."""
    return family == "metaworld"


def register_environment(env_id: str, factory: EnvFactory) -> None:
    """
    Register a new environment.

    Args:
        env_id: Unique environment identifier
        factory: Callable that returns a RoboticsEnvironment
    """
    if env_id in ENVIRONMENTS:
        raise ValueError(f"Environment {env_id} is already registered")
    ENVIRONMENTS[env_id] = factory
