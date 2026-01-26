from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


def _to_list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, Sequence):
        return list(value)
    return value


class CanonicalObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ee_pos_m: list[float] = Field(..., min_length=3, max_length=3)
    ee_quat_xyzw: list[float] = Field(..., min_length=4, max_length=4)
    ee_lin_vel_mps: list[float] = Field(..., min_length=3, max_length=3)
    ee_ang_vel_rps: list[float] = Field(..., min_length=3, max_length=3)
    gripper_01: float = Field(..., ge=0.0, le=1.0)
    rgb: dict[str, list] = Field(default_factory=dict)
    depth: list | None = None
    cam_intrinsics_K: list[list[float]] | None = None
    cam_extrinsics_T_world_cam: list[list[float]] | None = None

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
            return {k: _to_list(v) for k, v in value.items()}
        return value

    @field_validator("depth", "cam_intrinsics_K", "cam_extrinsics_T_world_cam", mode="before")
    @classmethod
    def _coerce_matrix(cls, value):
        return _to_list(value)

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
            img = np.array(self.rgb[cam_name], dtype=np.float32)
            image_arrays.append((img / 255.0).flatten())
        if image_arrays:
            return np.concatenate([proprio, *image_arrays]).astype(np.float32)
        return proprio

    def without_images(self) -> "CanonicalObservation":
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
    def from_array(cls, array: Any, gripper_default: float = 0.0) -> "CanonicalAction":
        arr = np.array(array, dtype=np.float32).flatten()
        twist = np.zeros(6, dtype=np.float32)
        gripper = gripper_default

        if arr.size == 4:
            twist[:3] = arr[:3]
            gripper = float(np.clip((arr[3] + 1.0) / 2.0, 0.0, 1.0))
        else:
            if arr.size:
                twist[: min(6, arr.size)] = arr[: min(6, arr.size)]
            if arr.size > 6:
                gripper = float(arr[6])

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
