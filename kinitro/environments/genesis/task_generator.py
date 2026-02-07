"""Scene-grounded task prompt generation for Genesis environments."""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import structlog

from kinitro.environments.genesis.robot_config import RobotConfig
from kinitro.environments.genesis.task_types import (
    SceneObject,
    TaskSpec,
    TaskType,
    check_task_feasibility,
)

T = TypeVar("T")

logger = structlog.get_logger()


# Templates for generating natural language prompts
PROMPT_TEMPLATES: dict[TaskType, list[str]] = {
    TaskType.NAVIGATE: [
        "Walk to the {color} {object}.",
        "Go to the {color} {object}.",
        "Move towards the {color} {object}.",
        "Navigate to the {color} {object}.",
        "Approach the {color} {object}.",
    ],
    TaskType.PICKUP: [
        "Pick up the {color} {object}.",
        "Grab the {color} {object}.",
        "Lift the {color} {object}.",
        "Take the {color} {object}.",
        "Get the {color} {object}.",
    ],
    TaskType.PLACE: [
        "Place the {color} {object} near the {dest_color} {dest_object}.",
        "Put the {color} {object} next to the {dest_color} {dest_object}.",
        "Set the {color} {object} by the {dest_color} {dest_object}.",
        "Move the {color} {object} to the {dest_color} {dest_object}.",
    ],
    TaskType.PUSH: [
        "Push the {color} {object} towards the {dest_color} {dest_object}.",
        "Shove the {color} {object} to the {dest_color} {dest_object}.",
        "Nudge the {color} {object} towards the {dest_color} {dest_object}.",
    ],
}


def _random_choice(items: list[T], rng: np.random.Generator) -> T | None:  # noqa: UP047
    """Select a random item from a list using the generator."""
    if not items:
        return None
    idx = int(rng.integers(0, len(items)))
    return items[idx]


def _get_pickupable_objects(objects: list[SceneObject]) -> list[SceneObject]:
    """Filter for objects that can be picked up."""
    return [obj for obj in objects if obj.pickupable and not obj.is_picked_up]


def _get_landmark_objects(objects: list[SceneObject]) -> list[SceneObject]:
    """Filter for large landmark objects (not pickupable)."""
    return [obj for obj in objects if not obj.pickupable]


class TaskGenerator:
    """Generates scene-grounded tasks for Genesis environments.

    This generator:
    - Analyzes scene objects to find feasible tasks
    - Generates natural language prompts grounded to actual objects
    - Validates task feasibility before returning
    - Filters by robot capabilities
    """

    def __init__(
        self,
        task_types: list[TaskType] | None = None,
        max_attempts: int = 20,
    ) -> None:
        self._task_types = task_types or list(TaskType)
        self._max_attempts = max_attempts

    def generate_task(
        self,
        objects: list[SceneObject],
        rng: np.random.Generator,
        robot_config: RobotConfig | None = None,
        task_type: TaskType | None = None,
    ) -> TaskSpec | None:
        """Generate a feasible task for the given scene.

        Args:
            objects: List of objects in the scene
            rng: Random number generator
            robot_config: Robot configuration (for capability filtering)
            task_type: Specific task type, or None for random

        Returns:
            TaskSpec if a feasible task was found, None otherwise
        """
        # Filter task types by robot capabilities
        available_types = self._task_types
        if robot_config is not None and robot_config.supported_task_types:
            available_types = [
                t for t in self._task_types if t.value in robot_config.supported_task_types
            ]

        if not available_types:
            logger.warning(
                "no_available_task_types", robot=robot_config.name if robot_config else None
            )
            return None

        if task_type is None:
            task_type = _random_choice(available_types, rng)

        for _ in range(self._max_attempts):
            if task_type is None:
                return None
            task = self._try_generate_task(objects, rng, task_type, robot_config)
            if task is not None:
                return task

            # Try a different task type
            task_type = _random_choice(available_types, rng)

        logger.warning(
            "task_generation_failed",
            num_objects=len(objects),
            max_attempts=self._max_attempts,
        )
        return None

    def _try_generate_task(
        self,
        objects: list[SceneObject],
        rng: np.random.Generator,
        task_type: TaskType,
        robot_config: RobotConfig | None,
    ) -> TaskSpec | None:
        """Attempt to generate a single task of the specified type."""
        supported = robot_config.supported_task_types if robot_config else None

        if task_type == TaskType.NAVIGATE:
            return self._generate_navigate_task(objects, rng, supported)
        elif task_type == TaskType.PICKUP:
            return self._generate_pickup_task(objects, rng, supported)
        elif task_type == TaskType.PLACE:
            return self._generate_place_task(objects, rng, supported)
        elif task_type == TaskType.PUSH:
            return self._generate_push_task(objects, rng, supported)
        return None

    def _generate_navigate_task(
        self,
        objects: list[SceneObject],
        rng: np.random.Generator,
        supported: list[str] | None,
    ) -> TaskSpec | None:
        """Generate a navigation task."""
        if not objects:
            return None

        target = _random_choice(objects, rng)
        if target is None:
            return None

        is_feasible, reason = check_task_feasibility(
            TaskType.NAVIGATE, target, robot_supported_tasks=supported
        )
        if not is_feasible:
            return None

        prompt = self._generate_prompt(TaskType.NAVIGATE, target, rng)

        return TaskSpec(
            task_type=TaskType.NAVIGATE,
            task_prompt=prompt,
            target_object_id=target.object_id,
            target_object_type=target.object_type,
            target_position=target.position,
            initial_state={"robot_start_pos": [0.0, 0.0, 0.75]},
        )

    def _generate_pickup_task(
        self,
        objects: list[SceneObject],
        rng: np.random.Generator,
        supported: list[str] | None,
    ) -> TaskSpec | None:
        """Generate a pickup task."""
        candidates = _get_pickupable_objects(objects)
        if not candidates:
            return None

        target = _random_choice(candidates, rng)
        if target is None:
            return None

        is_feasible, reason = check_task_feasibility(
            TaskType.PICKUP, target, robot_supported_tasks=supported
        )
        if not is_feasible:
            return None

        prompt = self._generate_prompt(TaskType.PICKUP, target, rng)

        return TaskSpec(
            task_type=TaskType.PICKUP,
            task_prompt=prompt,
            target_object_id=target.object_id,
            target_object_type=target.object_type,
            target_position=target.position,
            initial_state={"initial_height": target.position[2]},
        )

    def _generate_place_task(
        self,
        objects: list[SceneObject],
        rng: np.random.Generator,
        supported: list[str] | None,
    ) -> TaskSpec | None:
        """Generate a place task (pick up object, place near destination)."""
        pickupables = _get_pickupable_objects(objects)
        destinations = [obj for obj in objects if not obj.pickupable]

        if not pickupables or not destinations:
            return None

        target = _random_choice(pickupables, rng)
        destination = _random_choice(destinations, rng)
        if target is None or destination is None:
            return None

        is_feasible, reason = check_task_feasibility(
            TaskType.PLACE, target, destination, robot_supported_tasks=supported
        )
        if not is_feasible:
            return None

        prompt = self._generate_prompt(TaskType.PLACE, target, rng, destination)

        return TaskSpec(
            task_type=TaskType.PLACE,
            task_prompt=prompt,
            target_object_id=target.object_id,
            target_object_type=target.object_type,
            target_position=target.position,
            destination_object_id=destination.object_id,
            destination_position=destination.position,
            initial_state={"initial_target_pos": target.position},
        )

    def _generate_push_task(
        self,
        objects: list[SceneObject],
        rng: np.random.Generator,
        supported: list[str] | None,
    ) -> TaskSpec | None:
        """Generate a push task."""
        # Any non-picked-up object can be pushed; destination is a different object
        pushable = [obj for obj in objects if not obj.is_picked_up]
        if len(pushable) < 2:
            return None

        target = _random_choice(pushable, rng)
        if target is None:
            return None

        # Destination must be a different object
        dest_candidates = [obj for obj in pushable if obj.object_id != target.object_id]
        destination = _random_choice(dest_candidates, rng)
        if destination is None:
            return None

        is_feasible, reason = check_task_feasibility(
            TaskType.PUSH, target, destination, robot_supported_tasks=supported
        )
        if not is_feasible:
            return None

        prompt = self._generate_prompt(TaskType.PUSH, target, rng, destination)

        return TaskSpec(
            task_type=TaskType.PUSH,
            task_prompt=prompt,
            target_object_id=target.object_id,
            target_object_type=target.object_type,
            target_position=target.position,
            destination_object_id=destination.object_id,
            destination_position=destination.position,
            initial_state={"initial_target_pos": target.position},
        )

    def _generate_prompt(
        self,
        task_type: TaskType,
        target: SceneObject,
        rng: np.random.Generator,
        destination: SceneObject | None = None,
    ) -> str:
        """Generate a natural language prompt for the task."""
        templates = PROMPT_TEMPLATES[task_type]
        template = _random_choice(templates, rng)
        if template is None:
            template = templates[0]

        if destination is not None:
            return template.format(
                color=target.color,
                object=target.object_type,
                dest_color=destination.color,
                dest_object=destination.object_type,
            )

        return template.format(color=target.color, object=target.object_type)
