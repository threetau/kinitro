from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator


def _to_list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, Sequence):
        return list(value)
    return value


def _encode_image(img: np.ndarray) -> dict[str, Any]:
    """Encode image as base64 with metadata for efficient serialization."""
    return {
        "data": base64.b64encode(img.tobytes()).decode("ascii"),
        "shape": list(img.shape),
        "dtype": str(img.dtype),
    }


def _decode_image(encoded: dict[str, Any] | list) -> np.ndarray:
    """Decode image from base64 or nested list format."""
    if isinstance(encoded, list):
        # Legacy nested list format
        return np.array(encoded, dtype=np.uint8)
    # Base64 encoded format
    data = base64.b64decode(encoded["data"])
    shape = tuple(encoded["shape"])
    dtype = np.dtype(encoded["dtype"])
    return np.frombuffer(data, dtype=dtype).reshape(shape)


class CanonicalObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ee_pos_m: list[float] = Field(..., min_length=3, max_length=3)
    ee_quat_xyzw: list[float] = Field(..., min_length=4, max_length=4)
    ee_lin_vel_mps: list[float] = Field(..., min_length=3, max_length=3)
    ee_ang_vel_rps: list[float] = Field(..., min_length=3, max_length=3)
    gripper_01: float = Field(..., ge=0.0, le=1.0)
    # Images stored as base64-encoded dicts or nested lists (for backward compat)
    rgb: dict[str, dict | list] = Field(default_factory=dict)
    depth: dict | list | None = None
    cam_intrinsics_K: list[list[float]] | None = None  # noqa: N815 (CV convention)
    cam_extrinsics_T_world_cam: list[list[float]] | None = None  # noqa: N815 (CV convention)
    # Internal storage for numpy arrays (not serialized)
    _rgb_arrays: dict[str, np.ndarray] = PrivateAttr(default_factory=dict)
    _depth_array: np.ndarray | None = PrivateAttr(default=None)

    @field_validator(
        "ee_pos_m",
        "ee_quat_xyzw",
        "ee_lin_vel_mps",
        "ee_ang_vel_rps",
        mode="before",
    )
    @classmethod
    def _coerce_vector(cls, value):
        return _to_list(value)

    @field_validator("rgb", mode="before")
    @classmethod
    def _coerce_rgb(cls, value):
        if value is None:
            return {}
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    # Encode numpy array as base64
                    result[k] = _encode_image(v)
                elif isinstance(v, dict) and "data" in v:
                    # Already encoded
                    result[k] = v
                else:
                    # Legacy list format - keep as is
                    result[k] = _to_list(v) if not isinstance(v, list) else v
            return result
        return value

    @field_validator("depth", mode="before")
    @classmethod
    def _coerce_depth(cls, value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return _encode_image(value)
        if isinstance(value, dict) and "data" in value:
            return value
        return _to_list(value)

    @field_validator("cam_intrinsics_K", "cam_extrinsics_T_world_cam", mode="before")
    @classmethod
    def _coerce_matrix(cls, value):
        return _to_list(value)

    @property
    def images(self) -> dict[str, np.ndarray]:
        """Get RGB images as numpy arrays."""
        result = {}
        for name, data in self.rgb.items():
            if isinstance(data, dict) and "data" in data:
                result[name] = _decode_image(data)
            elif isinstance(data, list):
                result[name] = np.array(data, dtype=np.uint8)
        return result

    @property
    def depth_array(self) -> np.ndarray | None:
        """Get depth image as numpy array."""
        if self.depth is None:
            return None
        if isinstance(self.depth, dict) and "data" in self.depth:
            return _decode_image(self.depth)
        return np.array(self.depth, dtype=np.float32)

    def proprio_array(self) -> np.ndarray:
        return np.array(
            [
                *self.ee_pos_m,
                *self.ee_quat_xyzw,
                *self.ee_lin_vel_mps,
                *self.ee_ang_vel_rps,
                self.gripper_01,
            ],
            dtype=np.float32,
        )

    def to_flat_array(self, include_images: bool = False) -> np.ndarray:
        proprio = self.proprio_array()
        if not include_images:
            return proprio
        image_arrays = []
        for cam_name in sorted(self.rgb.keys()):
            img = self.images.get(cam_name)
            if img is not None:
                image_arrays.append((img.astype(np.float32) / 255.0).flatten())
        if image_arrays:
            return np.concatenate([proprio, *image_arrays]).astype(np.float32)
        return proprio

    def without_images(self) -> CanonicalObservation:
        return self.model_copy(update={"rgb": {}})

    def to_payload(self, include_images: bool = True) -> dict[str, Any]:
        if include_images:
            return self.model_dump(mode="python")
        return self.without_images().model_dump(mode="python")


class CanonicalAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    twist_ee_norm: list[float] = Field(..., min_length=6, max_length=6)
    gripper_01: float = Field(..., ge=0.0, le=1.0)

    @field_validator("twist_ee_norm", mode="before")
    @classmethod
    def _coerce_twist(cls, value: Any) -> Any:
        return _to_list(value)

    def to_array(self) -> np.ndarray:
        return np.array([*self.twist_ee_norm, self.gripper_01], dtype=np.float32)

    @classmethod
    def from_array(cls, array: Any, gripper_default: float = 0.0) -> CanonicalAction:
        arr = np.array(array, dtype=np.float32).flatten()
        twist = np.zeros(6, dtype=np.float32)
        gripper = gripper_default

        if arr.size == 4:
            # MetaWorld format: [x, y, z, gripper] where gripper is in [-1, 1]
            twist[:3] = arr[:3]
            gripper = float(np.clip((arr[3] + 1.0) / 2.0, 0.0, 1.0))
        else:
            # Standard format: [twist(6), gripper(1)]
            if arr.size:
                twist[: min(6, arr.size)] = arr[: min(6, arr.size)]
            if arr.size > 6:
                # Gripper should be in [0, 1]; clip if out of range
                gripper = float(np.clip(arr[6], 0.0, 1.0))

        return cls(twist_ee_norm=twist.tolist(), gripper_01=float(gripper))


class CanonicalStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    obs: CanonicalObservation
    action: CanonicalAction


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm <= 0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quat / norm).astype(np.float32)


def coerce_action(action: Any, gripper_default: float = 0.0) -> CanonicalAction:
    if isinstance(action, CanonicalAction):
        return action
    if isinstance(action, dict):
        return CanonicalAction.model_validate(action)
    return CanonicalAction.from_array(action, gripper_default=gripper_default)
