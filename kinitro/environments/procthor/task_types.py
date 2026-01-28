"""Task type definitions for ProcTHOR environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskType(Enum):
    """Types of tasks the agent can be asked to perform."""

    PICKUP = "pickup"  # Pick up an object
    PLACE = "place"  # Place an object at a location
    OPEN = "open"  # Open an openable object (fridge, drawer, etc.)
    CLOSE = "close"  # Close an openable object
    TOGGLE_ON = "toggle_on"  # Turn on a toggleable object (lamp, TV, etc.)
    TOGGLE_OFF = "toggle_off"  # Turn off a toggleable object


# Object properties required for each task type
TASK_REQUIRED_PROPERTIES: dict[TaskType, list[str]] = {
    TaskType.PICKUP: ["pickupable"],
    TaskType.PLACE: [],  # Any receptacle works
    TaskType.OPEN: ["openable"],
    TaskType.CLOSE: ["openable"],
    TaskType.TOGGLE_ON: ["toggleable"],
    TaskType.TOGGLE_OFF: ["toggleable"],
}


@dataclass
class TaskSpec:
    """Specification for a task to be performed in the environment."""

    task_type: TaskType
    task_prompt: str
    target_object_id: str
    target_object_type: str

    # For PLACE tasks: where to place the object
    destination_object_id: str | None = None
    destination_object_type: str | None = None

    # For tracking initial state (to detect completion)
    initial_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for TaskConfig."""
        return {
            "task_type": self.task_type.value,
            "task_prompt": self.task_prompt,
            "target_object_id": self.target_object_id,
            "target_object_type": self.target_object_type,
            "destination_object_id": self.destination_object_id,
            "destination_object_type": self.destination_object_type,
            "initial_state": self.initial_state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskSpec:
        """Deserialize from dictionary."""
        return cls(
            task_type=TaskType(data["task_type"]),
            task_prompt=data["task_prompt"],
            target_object_id=data["target_object_id"],
            target_object_type=data["target_object_type"],
            destination_object_id=data.get("destination_object_id"),
            destination_object_type=data.get("destination_object_type"),
            initial_state=data.get("initial_state", {}),
        )


@dataclass
class SceneObject:
    """Represents an object in the scene with its properties."""

    object_id: str
    object_type: str
    position: dict[str, float]
    rotation: dict[str, float]

    # Object properties
    pickupable: bool = False
    openable: bool = False
    toggleable: bool = False
    receptacle: bool = False
    breakable: bool = False
    moveable: bool = False

    # Current state
    is_open: bool = False
    is_toggled: bool = False
    is_picked_up: bool = False
    is_broken: bool = False
    is_moving: bool = False

    # Containment
    parent_receptacles: list[str] = field(default_factory=list)
    visible: bool = True

    @classmethod
    def from_ai2thor_metadata(cls, obj: dict[str, Any]) -> SceneObject:
        """Create from AI2-THOR object metadata."""
        position = obj.get("position", {})
        rotation = obj.get("rotation", {})

        return cls(
            object_id=obj.get("objectId", ""),
            object_type=obj.get("objectType", ""),
            position={
                "x": position.get("x", 0.0),
                "y": position.get("y", 0.0),
                "z": position.get("z", 0.0),
            },
            rotation={
                "x": rotation.get("x", 0.0),
                "y": rotation.get("y", 0.0),
                "z": rotation.get("z", 0.0),
            },
            pickupable=obj.get("pickupable", False),
            openable=obj.get("openable", False),
            toggleable=obj.get("toggleable", False),
            receptacle=obj.get("receptacle", False),
            breakable=obj.get("breakable", False),
            moveable=obj.get("moveable", False),
            is_open=obj.get("isOpen", False),
            is_toggled=obj.get("isToggled", False),
            is_picked_up=obj.get("isPickedUp", False),
            is_broken=obj.get("isBroken", False),
            is_moving=obj.get("isMoving", False),
            parent_receptacles=obj.get("parentReceptacles", []) or [],
            visible=obj.get("visible", True),
        )

    def has_property(self, prop: str) -> bool:
        """Check if object has a specific property."""
        return getattr(self, prop, False)

    def is_accessible(self) -> bool:
        """Check if object is accessible (not inside closed container)."""
        # For now, consider all objects accessible
        # A more sophisticated check would verify parent receptacles are open
        return True


def check_task_feasibility(
    task_type: TaskType,
    target: SceneObject,
    destination: SceneObject | None = None,
) -> tuple[bool, str]:
    """
    Check if a task is feasible given the target object.

    Returns:
        Tuple of (is_feasible, reason)
    """
    required_props = TASK_REQUIRED_PROPERTIES[task_type]

    # Check required properties
    for prop in required_props:
        if not target.has_property(prop):
            return False, f"Object {target.object_type} is not {prop}"

    # Check accessibility
    if not target.is_accessible():
        return False, f"Object {target.object_type} is not accessible"

    # Task-specific checks
    if task_type == TaskType.OPEN:
        if target.is_open:
            return False, f"Object {target.object_type} is already open"

    if task_type == TaskType.CLOSE:
        if not target.is_open:
            return False, f"Object {target.object_type} is already closed"

    if task_type == TaskType.TOGGLE_ON:
        if target.is_toggled:
            return False, f"Object {target.object_type} is already on"

    if task_type == TaskType.TOGGLE_OFF:
        if not target.is_toggled:
            return False, f"Object {target.object_type} is already off"

    if task_type == TaskType.PICKUP:
        if target.is_picked_up:
            return False, f"Object {target.object_type} is already picked up"

    if task_type == TaskType.PLACE:
        if destination is None:
            return False, "Place task requires a destination"
        if not destination.receptacle:
            return False, f"Destination {destination.object_type} is not a receptacle"

    return True, "Task is feasible"
