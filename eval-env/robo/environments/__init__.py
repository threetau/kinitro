"""
Robotics environment registry and utilities.

Provides a unified interface for loading and interacting with various
robotics simulation environments (MetaWorld, DM Control, ManiSkill).
"""

from robo.environments.base import EpisodeResult, RoboticsEnvironment, TaskConfig
from robo.environments.registry import ENVIRONMENTS, get_all_environment_ids, get_environment

__all__ = [
    "RoboticsEnvironment",
    "TaskConfig",
    "EpisodeResult",
    "ENVIRONMENTS",
    "get_environment",
    "get_all_environment_ids",
]
