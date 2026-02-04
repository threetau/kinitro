"""ProcTHOR procedural environment for embodied AI tasks."""

# Apply patches to procthor library before importing anything else
from kinitro.environments.procthor.patches import apply_patches

apply_patches()

from kinitro.environments.procthor.environment import ProcTHOREnvironment  # noqa: E402
from kinitro.environments.procthor.task_types import TaskType  # noqa: E402

__all__ = ["ProcTHOREnvironment", "TaskType"]
