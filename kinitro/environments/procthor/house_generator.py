"""ProcTHOR house generation wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from kinitro.environments.procthor.task_types import SceneObject

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
        self._procthor = None

    def _ensure_procthor(self) -> None:
        """Lazy import of procthor to avoid import errors."""
        if self._procthor is not None:
            return

        try:
            import procthor

            self._procthor = procthor
        except ImportError as exc:
            raise ImportError(
                "procthor is required for procedural house generation. "
                "Install with: pip install procthor"
            ) from exc

    def generate_house(self, seed: int) -> dict[str, Any]:
        """
        Generate a procedural house from a seed.

        Args:
            seed: Random seed for reproducible generation

        Returns:
            House specification dictionary compatible with AI2-THOR
        """
        if self._use_cache and seed in self._house_cache:
            return self._house_cache[seed]

        self._ensure_procthor()

        rng = np.random.default_rng(seed)

        # Determine number of rooms
        num_rooms = int(rng.integers(*self._num_rooms_range))

        # Generate house using procthor
        try:
            from procthor.generation import HouseGenerator as PTHouseGenerator

            generator = PTHouseGenerator(seed=seed)
            house = generator.generate(num_rooms=num_rooms)
            house_dict = house.to_dict() if hasattr(house, "to_dict") else house
        except Exception as e:
            logger.warning(
                "procthor_generation_failed",
                seed=seed,
                error=str(e),
            )
            # Fall back to a simple house spec
            house_dict = self._generate_fallback_house(seed)

        if self._use_cache:
            self._house_cache[seed] = house_dict

        return house_dict

    def _generate_fallback_house(self, seed: int) -> dict[str, Any]:
        """
        Generate a simple fallback house specification.

        This is used when ProcTHOR generation fails or for testing.
        Uses pre-built iTHOR FloorPlans as fallback.
        """
        rng = np.random.default_rng(seed)

        # Use existing iTHOR floor plans as fallback
        # These are the kitchen (1-30), living room (201-230),
        # bedroom (301-330), bathroom (401-430) scenes
        room_types = ["Kitchen", "LivingRoom", "Bedroom", "Bathroom"]
        floor_plan_ranges = {
            "Kitchen": (1, 30),
            "LivingRoom": (201, 230),
            "Bedroom": (301, 330),
            "Bathroom": (401, 430),
        }

        # Pick a random room type and floor plan
        room_type = rng.choice(room_types)
        start, end = floor_plan_ranges[room_type]
        floor_plan_num = int(rng.integers(start, end + 1))

        return {
            "scene_name": f"FloorPlan{floor_plan_num}",
            "is_procedural": False,
            "room_type": room_type,
            "seed": seed,
        }

    def get_scene_name(self, house: dict[str, Any]) -> dict[str, Any] | str:
        """
        Get the scene name to pass to AI2-THOR controller.

        For procedural houses, this returns the house dict itself.
        For fallback houses, this returns the iTHOR scene name.
        """
        if house.get("is_procedural", True):
            # Procedural house - return the full spec
            return house
        else:
            # Fallback to iTHOR scene
            return house["scene_name"]

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
