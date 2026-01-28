"""Scene-grounded task prompt generation for ProcTHOR."""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import structlog

from kinitro.environments.procthor.house_generator import (
    get_openable_objects,
    get_pickupable_objects,
    get_receptacles,
    get_toggleable_objects,
)
from kinitro.environments.procthor.task_types import (
    SceneObject,
    TaskSpec,
    TaskType,
    check_task_feasibility,
)

logger = structlog.get_logger()

T = TypeVar("T")


# Templates for generating natural language prompts
PROMPT_TEMPLATES: dict[TaskType, list[str]] = {
    TaskType.PICKUP: [
        "Pick up the {object}.",
        "Grab the {object}.",
        "Take the {object}.",
        "Get the {object}.",
        "Retrieve the {object}.",
    ],
    TaskType.PLACE: [
        "Put the {object} on the {destination}.",
        "Place the {object} on the {destination}.",
        "Set the {object} on the {destination}.",
        "Move the {object} to the {destination}.",
        "Put the {object} in the {destination}.",
    ],
    TaskType.OPEN: [
        "Open the {object}.",
        "Pull open the {object}.",
    ],
    TaskType.CLOSE: [
        "Close the {object}.",
        "Shut the {object}.",
    ],
    TaskType.TOGGLE_ON: [
        "Turn on the {object}.",
        "Switch on the {object}.",
        "Activate the {object}.",
    ],
    TaskType.TOGGLE_OFF: [
        "Turn off the {object}.",
        "Switch off the {object}.",
        "Deactivate the {object}.",
    ],
}


def format_object_name(object_type: str) -> str:
    """
    Convert AI2-THOR object type to human-readable name.

    E.g., "CoffeeMachine" -> "coffee machine"
    """
    # Insert space before capital letters and lowercase
    result = []
    for i, char in enumerate(object_type):
        if i > 0 and char.isupper():
            result.append(" ")
        result.append(char.lower())
    return "".join(result)


def _random_choice(items: list[T], rng: np.random.Generator) -> T | None:
    """Select a random item from a list using the generator."""
    if not items:
        return None
    idx = int(rng.integers(0, len(items)))
    return items[idx]


class TaskGenerator:
    """
    Generates scene-grounded tasks for ProcTHOR environments.

    This generator:
    - Analyzes scene objects to find feasible tasks
    - Generates natural language prompts
    - Validates task feasibility before returning
    """

    def __init__(
        self,
        task_types: list[TaskType] | None = None,
        max_attempts: int = 20,
    ) -> None:
        """
        Initialize the task generator.

        Args:
            task_types: Which task types to generate. None = all types.
            max_attempts: Max attempts to find a feasible task
        """
        self._task_types = task_types or list(TaskType)
        self._max_attempts = max_attempts

    def generate_task(
        self,
        objects: list[SceneObject],
        rng: np.random.Generator,
        task_type: TaskType | None = None,
    ) -> TaskSpec | None:
        """
        Generate a feasible task for the given scene.

        Args:
            objects: List of objects in the scene
            rng: Random number generator
            task_type: Specific task type to generate, or None for random

        Returns:
            TaskSpec if a feasible task was found, None otherwise
        """
        if task_type is None:
            task_type = _random_choice(self._task_types, rng)

        for _ in range(self._max_attempts):
            if task_type is None:
                return None
            task = self._try_generate_task(objects, rng, task_type)
            if task is not None:
                return task

            # Try a different task type
            task_type = _random_choice(self._task_types, rng)

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
    ) -> TaskSpec | None:
        """Attempt to generate a single task of the specified type."""
        if task_type == TaskType.PICKUP:
            return self._generate_pickup_task(objects, rng)
        elif task_type == TaskType.PLACE:
            return self._generate_place_task(objects, rng)
        elif task_type == TaskType.OPEN:
            return self._generate_open_task(objects, rng)
        elif task_type == TaskType.CLOSE:
            return self._generate_close_task(objects, rng)
        elif task_type == TaskType.TOGGLE_ON:
            return self._generate_toggle_on_task(objects, rng)
        elif task_type == TaskType.TOGGLE_OFF:
            return self._generate_toggle_off_task(objects, rng)
        return None

    def _generate_pickup_task(
        self,
        objects: list[SceneObject],
        rng: np.random.Generator,
    ) -> TaskSpec | None:
        """Generate a pickup task."""
        candidates = get_pickupable_objects(objects)
        candidates = [obj for obj in candidates if not obj.is_picked_up]

        if not candidates:
            return None

        target = _random_choice(candidates, rng)
        if target is None:
            return None

        is_feasible, reason = check_task_feasibility(TaskType.PICKUP, target)

        if not is_feasible:
            logger.debug("pickup_task_not_feasible", reason=reason)
            return None

        prompt = self._generate_prompt(TaskType.PICKUP, target, rng)

        return TaskSpec(
            task_type=TaskType.PICKUP,
            task_prompt=prompt,
            target_object_id=target.object_id,
            target_object_type=target.object_type,
            initial_state={"is_picked_up": target.is_picked_up},
        )

    def _generate_place_task(
        self,
        objects: list[SceneObject],
        rng: np.random.Generator,
    ) -> TaskSpec | None:
        """Generate a place task."""
        # For place tasks, we need a pickupable object and a receptacle
        pickupables = get_pickupable_objects(objects)
        receptacles = get_receptacles(objects)

        if not pickupables or not receptacles:
            return None

        target = _random_choice(pickupables, rng)
        destination = _random_choice(receptacles, rng)

        if target is None or destination is None:
            return None

        # Don't place object on itself or where it already is
        if target.object_id == destination.object_id:
            return None
        if destination.object_id in target.parent_receptacles:
            return None

        is_feasible, reason = check_task_feasibility(TaskType.PLACE, target, destination)

        if not is_feasible:
            logger.debug("place_task_not_feasible", reason=reason)
            return None

        prompt = self._generate_prompt(TaskType.PLACE, target, rng, destination)

        return TaskSpec(
            task_type=TaskType.PLACE,
            task_prompt=prompt,
            target_object_id=target.object_id,
            target_object_type=target.object_type,
            destination_object_id=destination.object_id,
            destination_object_type=destination.object_type,
            initial_state={
                "parent_receptacles": target.parent_receptacles.copy(),
            },
        )

    def _generate_open_task(
        self,
        objects: list[SceneObject],
        rng: np.random.Generator,
    ) -> TaskSpec | None:
        """Generate an open task."""
        candidates = get_openable_objects(objects)
        candidates = [obj for obj in candidates if not obj.is_open]

        if not candidates:
            return None

        target = _random_choice(candidates, rng)
        if target is None:
            return None

        is_feasible, reason = check_task_feasibility(TaskType.OPEN, target)

        if not is_feasible:
            logger.debug("open_task_not_feasible", reason=reason)
            return None

        prompt = self._generate_prompt(TaskType.OPEN, target, rng)

        return TaskSpec(
            task_type=TaskType.OPEN,
            task_prompt=prompt,
            target_object_id=target.object_id,
            target_object_type=target.object_type,
            initial_state={"is_open": target.is_open},
        )

    def _generate_close_task(
        self,
        objects: list[SceneObject],
        rng: np.random.Generator,
    ) -> TaskSpec | None:
        """Generate a close task."""
        candidates = get_openable_objects(objects)
        candidates = [obj for obj in candidates if obj.is_open]

        if not candidates:
            return None

        target = _random_choice(candidates, rng)
        if target is None:
            return None

        is_feasible, reason = check_task_feasibility(TaskType.CLOSE, target)

        if not is_feasible:
            logger.debug("close_task_not_feasible", reason=reason)
            return None

        prompt = self._generate_prompt(TaskType.CLOSE, target, rng)

        return TaskSpec(
            task_type=TaskType.CLOSE,
            task_prompt=prompt,
            target_object_id=target.object_id,
            target_object_type=target.object_type,
            initial_state={"is_open": target.is_open},
        )

    def _generate_toggle_on_task(
        self,
        objects: list[SceneObject],
        rng: np.random.Generator,
    ) -> TaskSpec | None:
        """Generate a toggle on task."""
        candidates = get_toggleable_objects(objects)
        candidates = [obj for obj in candidates if not obj.is_toggled]

        if not candidates:
            return None

        target = _random_choice(candidates, rng)
        if target is None:
            return None

        is_feasible, reason = check_task_feasibility(TaskType.TOGGLE_ON, target)

        if not is_feasible:
            logger.debug("toggle_on_task_not_feasible", reason=reason)
            return None

        prompt = self._generate_prompt(TaskType.TOGGLE_ON, target, rng)

        return TaskSpec(
            task_type=TaskType.TOGGLE_ON,
            task_prompt=prompt,
            target_object_id=target.object_id,
            target_object_type=target.object_type,
            initial_state={"is_toggled": target.is_toggled},
        )

    def _generate_toggle_off_task(
        self,
        objects: list[SceneObject],
        rng: np.random.Generator,
    ) -> TaskSpec | None:
        """Generate a toggle off task."""
        candidates = get_toggleable_objects(objects)
        candidates = [obj for obj in candidates if obj.is_toggled]

        if not candidates:
            return None

        target = _random_choice(candidates, rng)
        if target is None:
            return None

        is_feasible, reason = check_task_feasibility(TaskType.TOGGLE_OFF, target)

        if not is_feasible:
            logger.debug("toggle_off_task_not_feasible", reason=reason)
            return None

        prompt = self._generate_prompt(TaskType.TOGGLE_OFF, target, rng)

        return TaskSpec(
            task_type=TaskType.TOGGLE_OFF,
            task_prompt=prompt,
            target_object_id=target.object_id,
            target_object_type=target.object_type,
            initial_state={"is_toggled": target.is_toggled},
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

        object_name = format_object_name(target.object_type)

        if destination is not None:
            destination_name = format_object_name(destination.object_type)
            return template.format(object=object_name, destination=destination_name)

        return template.format(object=object_name)
