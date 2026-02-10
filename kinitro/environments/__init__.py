"""
Robotics environment registry and utilities.

Provides a unified interface for loading and interacting with various
robotics simulation environments (MetaWorld, Genesis, DM Control, ManiSkill).
"""

from kinitro.environments.base import EpisodeResult, RoboticsEnvironment, TaskConfig
from kinitro.environments.registry import (
    ENVIRONMENTS,
    get_all_environment_ids,
    get_environment,
    get_environments_by_family,
)

__all__ = [
    "RoboticsEnvironment",
    "TaskConfig",
    "EpisodeResult",
    "ENVIRONMENTS",
    "get_environment",
    "get_all_environment_ids",
    "get_environments_by_family",
]
