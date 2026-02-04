"""ProcTHOR house generation wrapper."""

from __future__ import annotations

import json
from typing import Any

import structlog

# Apply patches BEFORE importing procthor to fix known bugs
from kinitro.environments.procthor.patches import apply_patches

apply_patches()

from procthor.generation import HouseGenerator as PTHouseGenerator  # noqa: E402
from procthor.generation.room_specs import PROCTHOR10K_ROOM_SPEC_SAMPLER  # noqa: E402

from kinitro.environments.procthor.task_types import SceneObject  # noqa: E402

logger = structlog.get_logger()


class HouseGenerator:
    """
    Generates procedural houses using ProcTHOR.

    This wrapper handles:
    - Generating houses from seeds (reproducible)
    - Extracting scene metadata (rooms, objects, etc.)
    - Caching generated houses for efficiency
    """

    def __init__(
        self,
        num_rooms_range: tuple[int, int] = (3, 6),
        use_cache: bool = True,
    ) -> None:
        """
        Initialize the house generator.

        Args:
            num_rooms_range: (min, max) number of rooms in generated houses
            use_cache: Whether to cache generated houses
        """
        self._num_rooms_range = num_rooms_range
        self._use_cache = use_cache
        self._house_cache: dict[int, dict[str, Any]] = {}

    def generate_house(self, seed: int, controller: Any = None) -> dict[str, Any]:
        """
        Generate a procedural house from a seed using ProcTHOR.

        Args:
            seed: Random seed for reproducible generation
            controller: Optional AI2-THOR controller to use for generation.
                       If not provided, ProcTHOR will create its own.

        Returns:
            House specification dictionary compatible with AI2-THOR
        """
        if self._use_cache and seed in self._house_cache:
            return self._house_cache[seed]

        # Generate house using procthor
        try:
            # ProcTHOR requires a 'split' parameter and a room_spec_sampler
            # Use 'train' split and the default PROCTHOR10K room specs
            # Pass controller if provided to avoid creating a new one
            generator = PTHouseGenerator(
                split="train",
                seed=seed,
                room_spec_sampler=PROCTHOR10K_ROOM_SPEC_SAMPLER,
                controller=controller,
            )
            result = generator.sample()
            # sample() returns (House, Dict) tuple
            house = result[0] if isinstance(result, tuple) else result
            # House object has .data attribute which is the dict we need
            # (newer procthor versions don't have to_dict, but have .data)
            house_data = getattr(house, "data", None)
            if isinstance(house_data, dict):
                house_dict: dict[str, Any] = house_data
            else:
                to_dict = getattr(house, "to_dict", None)
                if callable(to_dict):
                    house_dict = to_dict()
                else:
                    to_json = getattr(house, "to_json", None)
                    if not callable(to_json):
                        raise RuntimeError(
                            "ProcTHOR house object has no supported serialization method"
                        )
                    house_dict = json.loads(to_json())
        except Exception as e:
            logger.warning(
                "procthor_generation_failed",
                seed=seed,
                error=str(e),
            )
            raise RuntimeError(f"ProcTHOR house generation failed: {e}") from e

        if self._use_cache:
            self._house_cache[seed] = house_dict

        return house_dict

    def get_scene_name(self, house: dict[str, Any]) -> dict[str, Any]:
        """
        Get the scene specification to pass to AI2-THOR controller.

        For ProcTHOR procedural houses, this returns the house dict itself.
        """
        return house

    def clear_cache(self) -> None:
        """Clear the house cache."""
        self._house_cache.clear()


def extract_scene_objects(
    event_metadata: dict[str, Any],
) -> list[SceneObject]:
    """
    Extract all objects from AI2-THOR event metadata.

    Args:
        event_metadata: Metadata dict from AI2-THOR event

    Returns:
        List of SceneObject instances
    """
    objects = []
    for obj_data in event_metadata.get("objects", []):
        try:
            obj = SceneObject.from_ai2thor_metadata(obj_data)
            objects.append(obj)
        except Exception as e:
            logger.debug(
                "failed_to_parse_object",
                object_id=obj_data.get("objectId"),
                error=str(e),
            )
    return objects


def get_objects_by_type(
    objects: list[SceneObject],
    object_type: str,
) -> list[SceneObject]:
    """Filter objects by type."""
    return [obj for obj in objects if obj.object_type == object_type]


def get_objects_by_property(
    objects: list[SceneObject],
    prop: str,
    value: bool = True,
) -> list[SceneObject]:
    """Filter objects by a boolean property."""
    return [obj for obj in objects if getattr(obj, prop, False) == value]


def get_pickupable_objects(objects: list[SceneObject]) -> list[SceneObject]:
    """Get all pickupable objects."""
    return get_objects_by_property(objects, "pickupable", True)


def get_openable_objects(objects: list[SceneObject]) -> list[SceneObject]:
    """Get all openable objects."""
    return get_objects_by_property(objects, "openable", True)


def get_toggleable_objects(objects: list[SceneObject]) -> list[SceneObject]:
    """Get all toggleable objects."""
    return get_objects_by_property(objects, "toggleable", True)


def get_receptacles(objects: list[SceneObject]) -> list[SceneObject]:
    """Get all receptacle objects."""
    return get_objects_by_property(objects, "receptacle", True)
