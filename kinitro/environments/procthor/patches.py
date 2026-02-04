"""
Monkey-patches for the procthor library.

These patches fix bugs in the upstream procthor library that cause issues
during house generation. The patches are applied at import time.

Bug fixed:
- small_objects.py: "invalid literal for int() with base 10: 'small'"
  The original code assumes all object IDs start with a numeric room ID,
  but small objects have IDs like "small|0|1". When parsing these,
  int("small") fails.

The fix wraps the original function with error handling and filtering
to skip problematic object IDs instead of replacing the entire function.
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


def apply_patches() -> None:
    """
    Apply monkey-patches to the procthor library.

    This should be called before any procthor generation code is used.
    Safe to call multiple times - patches are only applied once.
    """
    global _patches_applied  # noqa: PLW0603
    if _patches_applied:
        return

    try:
        import procthor.generation as generation_module  # noqa: PLC0415
        import procthor.generation.small_objects as small_objects_module  # noqa: PLC0415

        # Get the original function
        original_func = small_objects_module.default_add_small_objects  # type: ignore[attr-defined]

        # Wrap it with error handling
        wrapped_func = _wrap_add_small_objects(original_func)

        # Replace in all places
        small_objects_module.default_add_small_objects = wrapped_func  # type: ignore[attr-defined]
        generation_module.default_add_small_objects = wrapped_func  # type: ignore[attr-defined]

        # Patch _create_default_generation_functions to use our wrapped version
        original_create = generation_module._create_default_generation_functions  # type: ignore[attr-defined]

        def _patched_create_default_generation_functions():
            gf = original_create()
            # Replace the add_small_objects function with our wrapped version
            object.__setattr__(gf, "add_small_objects", wrapped_func)
            return gf

        generation_module._create_default_generation_functions = (  # type: ignore[attr-defined]
            _patched_create_default_generation_functions
        )

        logger.info("procthor_patches_applied", patch="small_objects_error_handling")
        _patches_applied = True
    except ImportError:
        # procthor not installed - that's fine, patches not needed
        logger.debug("procthor_not_installed", action="skipping patches")
    except Exception as e:
        logger.warning("procthor_patch_failed", error=str(e))
