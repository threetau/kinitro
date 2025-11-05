from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from .envs import EnvSpec

logger = logging.getLogger(__name__)


def to_numpy(value: Any) -> np.ndarray:
    """Best-effort conversion of supported types to contiguous numpy arrays."""
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)

    if arr.dtype == object:
        raise TypeError("Cannot convert object-dtype observation value to numpy array")

    return np.ascontiguousarray(arr)


def normalize_for_rpc(obs: Any) -> Any:
    """Prepare observation payload (array or dict of arrays) for RPC transmission."""
    if isinstance(obs, dict):
        normalized: dict[str, np.ndarray] = {}
        for key, value in obs.items():
            try:
                normalized[key] = to_numpy(value)
            except Exception as exc:
                logger.warning(
                    "Dropping observation key %s due to conversion failure: %s",
                    key,
                    exc,
                )
        return normalized

    return to_numpy(obs)


def extract_image_payloads(
    obs: Any,
    env_spec: EnvSpec,
) -> list[tuple[np.ndarray, str]]:
    """Extract HWC uint8 image payloads from observation dictionaries for logging."""
    if not isinstance(obs, dict):
        return []

    camera_names = list(getattr(env_spec, "camera_names", ()) or ("default",))
    image_entries: list[tuple[int, Any]] = []
    for key, value in obs.items():
        if not isinstance(key, str) or not key.startswith("observation.image"):
            continue
        suffix = key[len("observation.image") :]
        if suffix == "":
            index = 0
        else:
            try:
                index = max(int(suffix) - 1, 0)
            except ValueError:
                index = len(image_entries)
        image_entries.append((index, value))

    if not image_entries:
        return []

    image_payloads: list[tuple[np.ndarray, str]] = []
    image_entries.sort(key=lambda item: item[0])
    for idx, value in image_entries:
        try:
            img_arr = to_numpy(value)
        except Exception as exc:
            logger.warning("Failed to convert observation image %s: %s", idx, exc)
            continue

        if img_arr.ndim == 3 and img_arr.shape[0] in (1, 3):
            img_hwc = np.moveaxis(img_arr, 0, -1)
        else:
            img_hwc = np.asarray(img_arr)

        img_hwc = np.ascontiguousarray(img_hwc)
        if img_hwc.dtype != np.uint8:
            img_hwc = np.clip(img_hwc, 0, 255).astype(np.uint8)

        if camera_names and idx < len(camera_names):
            camera_name = camera_names[idx]
        elif camera_names:
            camera_name = camera_names[min(idx, len(camera_names) - 1)]
        else:
            camera_name = f"camera_{idx}"

        image_payloads.append((img_hwc, camera_name))

    return image_payloads


def coerce_success_flag(value: Any) -> bool:
    """Normalize various success flag representations to a boolean."""
    if isinstance(value, (bool, int, float)):
        return bool(value)
    if isinstance(value, np.generic):
        return bool(value.item())
    return bool(value)


def extract_success_flag(info_payload: Any) -> bool:
    """Extract a boolean success indicator from an info payload."""
    if not info_payload or not isinstance(info_payload, dict):
        return False
    if "success" not in info_payload:
        return False
    return coerce_success_flag(info_payload.get("success"))
