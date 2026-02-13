"""Task type definitions for Genesis environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from kinitro.types import FeasibilityResult, ObjectType


class TaskType(Enum):
    """Types of tasks the agent can be asked to perform."""

    NAVIGATE = "navigate"  # Walk to a target object
    PICKUP = "pickup"  # Pick up an object
    PLACE = "place"  # Place an object at a destination
    PUSH = "push"  # Push an object towards a destination


# Object properties required for each task type
TASK_REQUIRED_PROPERTIES: dict[TaskType, list[str]] = {
    TaskType.NAVIGATE: [],  # Any visible object works as target
    TaskType.PICKUP: ["pickupable"],
    TaskType.PLACE: ["pickupable"],  # Need to pick up first, then place
    TaskType.PUSH: [],  # Any moveable object
}


@dataclass
class TaskSpec:
    """Specification for a task to be performed in a Genesis environment."""

    task_type: TaskType
    task_prompt: str
    target_object_id: str
    target_object_type: ObjectType
    target_position: list[float]  # [x, y, z]

    # For PLACE/PUSH tasks: where to deliver the object
    destination_object_id: str | None = None
    destination_position: list[float] | None = None

    # Any: initial state values are heterogeneous (positions, flags, etc.)
    initial_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:  # Any: mixed value types after serialization
        """Serialize to dictionary for TaskConfig."""
        return {
            "task_type": self.task_type.value,
            "task_prompt": self.task_prompt,
            "target_object_id": self.target_object_id,
            "target_object_type": self.target_object_type.value,
            "target_position": self.target_position,
            "destination_object_id": self.destination_object_id,
            "destination_position": self.destination_position,
            "initial_state": self.initial_state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskSpec:  # Any: deserialized JSON values
        """Deserialize from dictionary."""
        return cls(
            task_type=TaskType(data["task_type"]),
            task_prompt=data["task_prompt"],
            target_object_id=data["target_object_id"],
            target_object_type=ObjectType(data["target_object_type"]),
            target_position=data["target_position"],
            destination_object_id=data.get("destination_object_id"),
            destination_position=data.get("destination_position"),
            initial_state=data.get("initial_state", {}),
        )


@dataclass
class SceneObject:
    """Represents a primitive object placed in a Genesis scene."""

    object_id: str
    object_type: ObjectType
    position: list[float]  # [x, y, z]
    color: str  # "red", "green", "blue", etc.
    color_rgb: tuple[float, float, float]
    size: float  # Characteristic dimension (half-extent or radius)
    pickupable: bool  # Small enough to grasp

    # Runtime state
    is_picked_up: bool = False


# Predefined color palette with distinct RGB values.
# TODO: extend to RGBA (alpha channel) for transparency effects (e.g. glass)
# once Genesis material/texture support is added.
OBJECT_COLORS: dict[str, tuple[float, float, float]] = {
    "red": (0.9, 0.2, 0.2),
    "green": (0.2, 0.8, 0.2),
    "blue": (0.2, 0.3, 0.9),
    "yellow": (0.9, 0.9, 0.2),
    "orange": (0.9, 0.5, 0.1),
    "purple": (0.6, 0.2, 0.8),
    "cyan": (0.2, 0.8, 0.8),
    "white": (0.9, 0.9, 0.9),
}

OBJECT_TYPES = [e.value for e in ObjectType]


def check_task_feasibility(
    task_type: TaskType,
    target: SceneObject,
    destination: SceneObject | None = None,
    robot_supported_tasks: list[TaskType] | None = None,
) -> FeasibilityResult:
    """
    Check if a task is feasible given the target object and robot capabilities.

    Returns:
        Tuple of (is_feasible, reason)
    """
    # Check robot supports this task type
    if robot_supported_tasks is not None:
        if task_type not in robot_supported_tasks:
            return FeasibilityResult(False, f"Robot does not support task type {task_type.value}")

    # Check required properties
    required_props = TASK_REQUIRED_PROPERTIES[task_type]
    for prop in required_props:
        if not getattr(target, prop, False):
            return FeasibilityResult(False, f"Object {target.object_type.value} is not {prop}")

    # Task-specific checks
    match task_type:
        case TaskType.NAVIGATE:
            pass  # Always feasible if target exists
        case TaskType.PICKUP:
            if target.is_picked_up:
                return FeasibilityResult(
                    False, f"Object {target.object_type.value} is already picked up"
                )
        case TaskType.PLACE:
            if destination is None:
                return FeasibilityResult(False, "Place task requires a destination")
        case TaskType.PUSH:
            if destination is None:
                return FeasibilityResult(False, "Push task requires a destination")
            if target.object_id == destination.object_id:
                return FeasibilityResult(False, "Cannot push object to itself")

    return FeasibilityResult(True, "Task is feasible")
