"""
Monkey-patches for the procthor library.

These patches fix bugs in the upstream procthor library that cause issues
during house generation. The patches are applied at import time.

Bugs fixed:
1. small_objects.py: "invalid literal for int() with base 10: 'small'"
   The original code assumes all object IDs start with a numeric room ID,
   but small objects have IDs like "small|0|1". When parsing these,
   int("small") fails.

2. objects.py AssetGroup.assets_dict: KeyError on parent_id lookup
   The assets_dict property builds a dict keyed by instanceId, then tries
   to look up parent objects using parentInstanceId. When these don't match
   (e.g., parent_id='2|0|0' not found), it raises KeyError.

The fixes wrap original functions with error handling to skip problematic
cases instead of crashing the entire house generation.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

import structlog

logger = structlog.get_logger(__name__)

_patches_applied = False


def _safe_int_parse(value: str) -> int | None:
    """
    Safely parse a string to int, returning None if it fails.

    This is used to handle object IDs like "small|0|1" where the prefix
    is not a number.
    """
    try:
        return int(value)
    except ValueError:
        return None


def _wrap_add_small_objects(original_func: Callable) -> Callable:
    """
    Wrap the original add_small_objects function with error handling.

    Instead of replacing the function entirely, we wrap it to:
    1. Catch the "invalid literal for int()" error
    2. Skip small object addition if it fails (house still works)
    3. Log a warning but don't fail the entire house generation
    """

    @functools.wraps(original_func)
    def wrapped(
        partial_house: Any,
        controller: Any,
        pt_db: Any,
        split: Any,
        rooms: dict[int, Any],
        max_object_types_per_room: int = 10000,
    ) -> None:
        try:
            return original_func(
                partial_house,
                controller,
                pt_db,
                split,
                rooms,
                max_object_types_per_room,
            )
        except ValueError as e:
            if "invalid literal for int()" in str(e):
                # This is the known bug - skip small object addition
                logger.warning(
                    "procthor_small_objects_skipped",
                    error=str(e),
                    hint="Known procthor bug with 'small|' object IDs - house generated without small objects",
                )
                return None
            raise

    return wrapped


def _patch_asset_group_sample_object_placement(
    asset_group_generator_class: type,
) -> None:
    """
    Patch AssetGroupGenerator.sample_object_placement to handle missing parent IDs.

    The original code does: parent_asset_lookup[asset["parentId"]]
    when processing child assets, but if the parent wasn't added to
    the lookup dict, it raises KeyError.

    This patch wraps the method to catch KeyError and skip the
    problematic asset group instead of failing the entire generation.
    """
    original_method = asset_group_generator_class.sample_object_placement  # type: ignore[attr-defined]

    def patched_sample_object_placement(
        self,
        allow_clipping: bool = True,
        floor_position: float = 0,
        use_thumbnail_assets: bool = False,
        chosen_asset_ids: dict | None = None,
    ):
        logger.debug(
            "procthor_sample_object_placement_called",
            asset_group=getattr(self, "name", "unknown"),
        )
        try:
            result = original_method(
                self,
                allow_clipping=allow_clipping,
                floor_position=floor_position,
                use_thumbnail_assets=use_thumbnail_assets,
                chosen_asset_ids=chosen_asset_ids,
            )
            return result
        except KeyError as e:
            logger.warning(
                "procthor_asset_group_generation_skipped",
                error=str(e),
                asset_group=getattr(self, "name", "unknown"),
                hint="Parent asset not found in lookup - skipping this asset group",
            )
            # Return an empty result so generation can continue
            return {
                "objects": [],
                "bounds": {
                    "x": {"min": 0, "max": 0},
                    "y": {"min": 0, "max": 0},
                    "z": {"min": 0, "max": 0},
                },
                "center": {"x": 0, "y": 0, "z": 0},
                "xLength": 0,
                "zLength": 0,
            }

    asset_group_generator_class.sample_object_placement = (  # type: ignore[method-assign]
        patched_sample_object_placement
    )


def _patch_asset_group_assets_dict(asset_group_class: type) -> None:
    """
    Patch the AssetGroup.assets_dict property to handle missing parent IDs.

    The original code does: objects[parent_id] where parent_id comes from
    parentInstanceId metadata, but the dict is keyed by instanceId.
    When these don't match (e.g., parent_id='2|0|0' not found), it raises KeyError.

    This patch replaces the parent-child linking logic with a safe version
    that skips invalid relationships instead of crashing.
    """

    @property
    def patched_assets_dict(self) -> list:
        from procthor.utils.types import Vector3  # noqa: PLC0415

        # Get asset group metadata for parent-child relationships
        asset_group_metadata = self.pt_db.ASSET_GROUPS[self.asset_group_name]["assetMetadata"]

        # Build parent-child pairs
        parent_children_pairs = []
        for child_id, metadata in asset_group_metadata.items():
            if (
                "parentInstanceId" in metadata
                and metadata["position"]["verticalAlignment"] == "above"
            ):
                parent_id = str(metadata["parentInstanceId"])
                parent_children_pairs.append((parent_id, child_id))

        # Build objects dict keyed by instanceId
        objects = {
            obj["instanceId"]: {
                "id": f"{self.room_id}|{self.object_n}|{i}",
                "position": obj["position"],
                "rotation": Vector3(x=0, y=obj["rotation"], z=0),
                "assetId": obj["assetId"],
                "kinematic": bool(
                    self.pt_db.PLACEMENT_ANNOTATIONS.loc[
                        self.pt_db.ASSET_ID_DATABASE[obj["assetId"]]["objectType"]
                    ]["isKinematic"]
                ),
            }
            for i, obj in enumerate(self.objects)
        }

        # Safely assign children to parents - skip if parent or child missing
        child_instance_ids = set()
        for parent_id, child_id in parent_children_pairs:
            # Check if both parent and child exist in objects
            if parent_id not in objects:
                logger.debug(
                    "procthor_parent_not_found",
                    parent_id=parent_id,
                    child_id=child_id,
                    asset_group=self.asset_group_name,
                )
                continue
            if child_id not in objects:
                logger.debug(
                    "procthor_child_not_found",
                    parent_id=parent_id,
                    child_id=child_id,
                    asset_group=self.asset_group_name,
                )
                continue

            # Both exist - create parent-child relationship
            if "children" not in objects[parent_id]:
                objects[parent_id]["children"] = []
            objects[parent_id]["children"].append(objects[child_id])  # type: ignore[union-attr]
            child_instance_ids.add(child_id)

        # Remove children that were assigned to parents
        for child_id in child_instance_ids:
            del objects[child_id]

        return list(objects.values())

    asset_group_class.assets_dict = patched_assets_dict  # type: ignore[method-assign, assignment]


def apply_patches() -> None:
    """
    Apply monkey-patches to the procthor library.

    This should be called before any procthor generation code is used.
    Safe to call multiple times - patches are only applied once.
    """
    global _patches_applied  # noqa: PLW0603
    if _patches_applied:
        return

    patches_applied_list = []

    try:
        import procthor.generation as generation_module  # noqa: PLC0415
        import procthor.generation.objects as objects_module  # noqa: PLC0415
        import procthor.generation.small_objects as small_objects_module  # noqa: PLC0415

        # Patch 1: small_objects error handling
        try:
            original_func = small_objects_module.default_add_small_objects  # type: ignore[attr-defined]
            wrapped_func = _wrap_add_small_objects(original_func)

            small_objects_module.default_add_small_objects = wrapped_func  # type: ignore[attr-defined]
            generation_module.default_add_small_objects = wrapped_func  # type: ignore[attr-defined]

            original_create = generation_module._create_default_generation_functions  # type: ignore[attr-defined]

            def _patched_create_default_generation_functions():
                gf = original_create()
                object.__setattr__(gf, "add_small_objects", wrapped_func)
                return gf

            generation_module._create_default_generation_functions = (  # type: ignore[attr-defined]
                _patched_create_default_generation_functions
            )
            patches_applied_list.append("small_objects_error_handling")
        except Exception as e:
            logger.warning("procthor_patch_small_objects_failed", error=str(e))

        # Patch 2: AssetGroup.assets_dict KeyError handling
        try:
            _patch_asset_group_assets_dict(objects_module.AssetGroup)
            patches_applied_list.append("asset_group_assets_dict")
        except Exception as e:
            logger.warning("procthor_patch_asset_group_failed", error=str(e))

        # Patch 3: AssetGroupGenerator.sample_object_placement KeyError handling
        try:
            import procthor.generation.asset_groups as asset_groups_module  # noqa: PLC0415

            _patch_asset_group_sample_object_placement(asset_groups_module.AssetGroupGenerator)
            patches_applied_list.append("asset_group_generator")
        except Exception as e:
            logger.warning("procthor_patch_asset_group_generator_failed", error=str(e))

        if patches_applied_list:
            logger.info("procthor_patches_applied", patches=patches_applied_list)
        _patches_applied = True
    except ImportError:
        # procthor not installed - that's fine, patches not needed
        logger.debug("procthor_not_installed", action="skipping patches")
    except Exception as e:
        logger.warning("procthor_patch_failed", error=str(e))
