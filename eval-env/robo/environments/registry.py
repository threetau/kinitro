"""Environment registry for loading robotics environments."""

from collections.abc import Callable

import structlog

from robo.environments.base import RoboticsEnvironment

logger = structlog.get_logger()

# Type alias for environment factory functions
EnvFactory = Callable[[], RoboticsEnvironment]

# Track which optional dependencies are available
_OPTIONAL_DEPS = {
    "dm_control": False,
    "maniskill": False,
}


def _check_optional_deps():
    """Check which optional environment dependencies are available."""
    global _OPTIONAL_DEPS

    try:
        from dm_control import suite  # noqa: F401

        _OPTIONAL_DEPS["dm_control"] = True
    except ImportError:
        pass

    try:
        import mani_skill2  # noqa: F401

        _OPTIONAL_DEPS["maniskill"] = True
    except ImportError:
        pass


# Check on module load
_check_optional_deps()


def _make_metaworld_env(task: str) -> EnvFactory:
    """Create factory for MetaWorld environment."""

    def factory() -> RoboticsEnvironment:
        from robo.environments.metaworld_env import MetaWorldEnvironment

        return MetaWorldEnvironment(task)

    return factory


def _make_dm_control_env(domain: str, task: str) -> EnvFactory:
    """Create factory for DM Control environment."""

    def factory() -> RoboticsEnvironment:
        if not _OPTIONAL_DEPS["dm_control"]:
            raise ImportError("dm_control is not installed. Install with: pip install dm_control")
        from robo.environments.dm_control_env import DMControlEnvironment

        return DMControlEnvironment(domain, task)

    return factory


def _make_maniskill_env(task: str) -> EnvFactory:
    """Create factory for ManiSkill environment."""

    def factory() -> RoboticsEnvironment:
        if not _OPTIONAL_DEPS["maniskill"]:
            raise ImportError("mani_skill2 is not installed. Install with: pip install mani-skill2")
        from robo.environments.maniskill_env import ManiSkillEnvironment

        return ManiSkillEnvironment(task)

    return factory


# ===========================================================================
# ENVIRONMENT REGISTRY
# ===========================================================================
# Core environments (always available)
# ===========================================================================

ENVIRONMENTS: dict[str, EnvFactory] = {
    # =========================================================================
    # MANIPULATION (MetaWorld V3) - Core environments, always available
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

# ===========================================================================
# OPTIONAL ENVIRONMENTS (registered if dependencies available)
# ===========================================================================

# DM Control - Locomotion (optional)
_DM_CONTROL_ENVS = {
    "dm_control/walker-stand": ("walker", "stand"),
    "dm_control/walker-walk": ("walker", "walk"),
    "dm_control/walker-run": ("walker", "run"),
    "dm_control/cheetah-run": ("cheetah", "run"),
    "dm_control/hopper-stand": ("hopper", "stand"),
    "dm_control/hopper-hop": ("hopper", "hop"),
    "dm_control/humanoid-stand": ("humanoid", "stand"),
    "dm_control/humanoid-walk": ("humanoid", "walk"),
}

# ManiSkill - Dexterous manipulation (optional)
_MANISKILL_ENVS = {
    "maniskill/PickCube-v1": "PickCube-v1",
    "maniskill/StackCube-v1": "StackCube-v1",
    "maniskill/PegInsertionSide-v1": "PegInsertionSide-v1",
    "maniskill/OpenCabinetDoor-v1": "OpenCabinetDoor-v1",
    "maniskill/OpenCabinetDrawer-v1": "OpenCabinetDrawer-v1",
}


def _register_optional_environments():
    """Register optional environments if their dependencies are available."""
    if _OPTIONAL_DEPS["dm_control"]:
        for env_id, (domain, task) in _DM_CONTROL_ENVS.items():
            ENVIRONMENTS[env_id] = _make_dm_control_env(domain, task)
        logger.info("registered_dm_control_environments", count=len(_DM_CONTROL_ENVS))

    if _OPTIONAL_DEPS["maniskill"]:
        for env_id, task in _MANISKILL_ENVS.items():
            ENVIRONMENTS[env_id] = _make_maniskill_env(task)
        logger.info("registered_maniskill_environments", count=len(_MANISKILL_ENVS))


# Register optional environments on module load
_register_optional_environments()


def get_environment(env_id: str) -> RoboticsEnvironment:
    """
    Load a robotics environment by ID.

    Args:
        env_id: Environment identifier (e.g., 'metaworld/pick-place-v2')

    Returns:
        Initialized RoboticsEnvironment instance

    Raises:
        ValueError: If environment ID is not found
        ImportError: If optional dependency is not installed
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
        family: Environment family ('metaworld', 'dm_control', 'maniskill')

    Returns:
        List of environment IDs in that family
    """
    return [env_id for env_id in ENVIRONMENTS if env_id.startswith(f"{family}/")]


def get_available_families() -> list[str]:
    """Get list of available environment families based on installed deps."""
    families = ["metaworld"]  # Always available
    if _OPTIONAL_DEPS["dm_control"]:
        families.append("dm_control")
    if _OPTIONAL_DEPS["maniskill"]:
        families.append("maniskill")
    return families


def is_family_available(family: str) -> bool:
    """Check if an environment family is available."""
    if family == "metaworld":
        return True
    return _OPTIONAL_DEPS.get(family, False)


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
