"""Procedural scene generation for Genesis environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from kinitro.environments.genesis.task_types import (
    OBJECT_COLORS,
    OBJECT_TYPES,
    SceneObject,
)

logger = structlog.get_logger()


@dataclass
class SceneObjectConfig:
    """Configuration for an object to be placed in the scene."""

    object_id: str
    object_type: str  # "box", "sphere", "cylinder"
    position: list[float]  # [x, y, z]
    color: str
    color_rgb: tuple[float, float, float]
    size: float  # half-extent or radius
    pickupable: bool


@dataclass
class SceneConfig:
    """Configuration for a procedurally generated scene."""

    terrain_type: str  # Currently always "flat"; kept for future scene types
    terrain_params: dict[str, Any] = field(default_factory=dict)
    objects: list[SceneObjectConfig] = field(default_factory=list)

    def get_scene_objects(self) -> list[SceneObject]:
        """Convert object configs to SceneObject instances."""
        return [
            SceneObject(
                object_id=obj.object_id,
                object_type=obj.object_type,
                position=obj.position,
                color=obj.color,
                color_rgb=obj.color_rgb,
                size=obj.size,
                pickupable=obj.pickupable,
            )
            for obj in self.objects
        ]


# Size ranges for objects
SMALL_OBJECT_SIZE = (0.03, 0.08)  # Pickupable
LARGE_OBJECT_SIZE = (0.1, 0.25)  # Landmarks, not pickupable


class SceneGenerator:
    """Generates procedural scenes with objects for Genesis environments.

    Scenes include:
    - Flat ground plane
    - 3-6 primitive objects (mix of small pickupable and large landmarks)
    - Distinct colors from a predefined palette
    """

    def __init__(
        self,
        num_objects: tuple[int, int] = (3, 6),
        arena_size: float = 5.0,
    ) -> None:
        self._num_objects_range = num_objects
        self._arena_size = arena_size

    def generate_scene(self, seed: int) -> SceneConfig:
        """Generate a procedural scene configuration from seed.

        Args:
            seed: Random seed for deterministic generation

        Returns:
            SceneConfig with terrain and object placement
        """
        rng = np.random.default_rng(seed)

        # Generate objects
        num_objects = int(rng.integers(self._num_objects_range[0], self._num_objects_range[1] + 1))
        objects = self._generate_objects(num_objects, rng)

        return SceneConfig(
            terrain_type="flat",
            terrain_params={},
            objects=objects,
        )

    def _generate_objects(
        self,
        num_objects: int,
        rng: np.random.Generator,
    ) -> list[SceneObjectConfig]:
        """Generate a set of objects with distinct colors."""
        objects = []
        available_colors = list(OBJECT_COLORS.keys())
        rng.shuffle(available_colors)

        # Ensure mix of pickupable and landmark objects
        # At least 1 pickupable and 1 landmark
        num_pickupable = max(1, int(rng.integers(1, max(2, num_objects - 1))))
        num_pickupable = min(num_pickupable, num_objects - 1)

        for i in range(num_objects):
            color_name = available_colors[i % len(available_colors)]
            color_rgb = OBJECT_COLORS[color_name]
            obj_type = rng.choice(OBJECT_TYPES)
            is_pickupable = i < num_pickupable

            if is_pickupable:
                size = float(rng.uniform(*SMALL_OBJECT_SIZE))
            else:
                size = float(rng.uniform(*LARGE_OBJECT_SIZE))

            # Place objects within arena, avoiding center (robot spawn)
            position = self._generate_object_position(rng, is_pickupable)

            objects.append(
                SceneObjectConfig(
                    object_id=f"obj_{i:02d}_{color_name}_{obj_type}",
                    object_type=obj_type,
                    position=position,
                    color=color_name,
                    color_rgb=color_rgb,
                    size=size,
                    pickupable=is_pickupable,
                )
            )

        return objects

    def _generate_object_position(
        self,
        rng: np.random.Generator,
        pickupable: bool,
    ) -> list[float]:
        """Generate a valid position for an object, avoiding robot spawn area."""
        half_arena = self._arena_size / 2.0
        min_dist_from_center = 0.8  # Avoid robot spawn area

        for _ in range(50):
            x = float(rng.uniform(-half_arena, half_arena))
            y = float(rng.uniform(-half_arena, half_arena))

            dist = np.sqrt(x**2 + y**2)
            if dist < min_dist_from_center:
                continue

            # Pickupable objects should be closer (reachable)
            if pickupable and dist > half_arena * 0.7:
                continue

            return [x, y, 0.05]  # Small offset above flat ground

        # Fallback position
        return [1.5, 0.0, 0.05]

    def build_scene(self, gs_scene: Any, scene_config: SceneConfig) -> list[Any]:
        """Materialize a SceneConfig into a Genesis scene.

        Args:
            gs_scene: Genesis scene object (gs.Scene)
            scene_config: The procedural scene configuration

        Returns:
            List of Genesis entity references for object tracking
        """
        import genesis as gs  # noqa: PLC0415

        entities = []

        # Add flat ground plane
        gs_scene.add_entity(gs.morphs.Plane())

        # Add objects (non-pickupable landmarks are fixed to prevent rolling)
        for obj_config in scene_config.objects:
            pos = tuple(obj_config.position)
            surface = gs.surfaces.Default(color=obj_config.color_rgb)
            is_fixed = not obj_config.pickupable

            if obj_config.object_type == "box":
                entity = gs_scene.add_entity(
                    gs.morphs.Box(
                        pos=pos,
                        size=(obj_config.size, obj_config.size, obj_config.size),
                        fixed=is_fixed,
                    ),
                    surface=surface,
                )
            elif obj_config.object_type == "sphere":
                entity = gs_scene.add_entity(
                    gs.morphs.Sphere(
                        pos=pos,
                        radius=obj_config.size,
                        fixed=is_fixed,
                    ),
                    surface=surface,
                )
            elif obj_config.object_type == "cylinder":
                entity = gs_scene.add_entity(
                    gs.morphs.Cylinder(
                        pos=pos,
                        radius=obj_config.size,
                        height=obj_config.size * 2,
                        fixed=is_fixed,
                    ),
                    surface=surface,
                )
            else:
                continue

            entities.append(entity)

        return entities
