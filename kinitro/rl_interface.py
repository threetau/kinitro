"""
Extensible RL interface for robotics environments.

This module provides the canonical observation and action interfaces
that work across different robot embodiments (manipulators, mobile robots,
navigation agents, etc.).

Key classes:
- Observation: Extensible observation with proprio dict, camera images, and extras
- Action: Extensible action with continuous and discrete channels

Conventional keys:
- ProprioKeys: Standard proprioceptive channel names
- ActionKeys: Standard action channel names
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

# =============================================================================
# Conventional Key Names
# =============================================================================


class ProprioKeys:
    """
    Conventional proprioceptive channel names.

    These are suggestions - environments can use any keys they want.
    Using these conventions improves interoperability between environments.
    """

    # End-effector state (single arm)
    EE_POS = "ee_pos"  # [x, y, z] meters
    EE_QUAT = "ee_quat"  # [x, y, z, w] quaternion
    EE_VEL_LIN = "ee_vel_lin"  # [vx, vy, vz] m/s
    EE_VEL_ANG = "ee_vel_ang"  # [wx, wy, wz] rad/s

    # Gripper
    GRIPPER = "gripper"  # [state] 0=open, 1=closed
    # or [f1, f2, ...] for multi-finger

    # Joint state
    JOINT_POS = "joint_pos"  # [j1, j2, ..., jN] radians
    JOINT_VEL = "joint_vel"  # [j1, j2, ..., jN] rad/s
    JOINT_TORQUE = "joint_torque"  # [t1, t2, ..., tN] Nm

    # Mobile base
    BASE_POS = "base_pos"  # [x, y, z] meters
    BASE_QUAT = "base_quat"  # [x, y, z, w] quaternion
    BASE_HEADING = "base_heading"  # [theta] radians (2D heading)
    BASE_VEL = "base_vel"  # [vx, vy, vtheta] or [vx, vy, vz, wx, wy, wz]

    # Dual arm (prefix with left_/right_)
    LEFT_EE_POS = "left_ee_pos"
    LEFT_EE_QUAT = "left_ee_quat"
    LEFT_GRIPPER = "left_gripper"
    RIGHT_EE_POS = "right_ee_pos"
    RIGHT_EE_QUAT = "right_ee_quat"
    RIGHT_GRIPPER = "right_gripper"

    # Sensing
    FORCE = "force"  # [fx, fy, fz] Newtons
    TORQUE = "torque"  # [tx, ty, tz] Nm
    IMU_ACC = "imu_acc"  # [ax, ay, az] m/s^2
    IMU_GYRO = "imu_gyro"  # [wx, wy, wz] rad/s


class ActionKeys:
    """Conventional action channel names."""

    # Cartesian control
    EE_TWIST = "ee_twist"  # [vx, vy, vz, wx, wy, wz] normalized
    EE_POS_TARGET = "ee_pos_target"  # [x, y, z] position target

    # Gripper
    GRIPPER = "gripper"  # [cmd] 0=open, 1=closed

    # Joint control
    JOINT_VEL = "joint_vel"  # [j1, j2, ..., jN] normalized
    JOINT_POS_TARGET = "joint_pos_target"  # [j1, ..., jN] position targets

    # Mobile base
    BASE_VEL = "base_vel"  # [vx, vy, vtheta] normalized

    # Dual arm
    LEFT_TWIST = "left_twist"
    LEFT_GRIPPER = "left_gripper"
    RIGHT_TWIST = "right_twist"
    RIGHT_GRIPPER = "right_gripper"


# =============================================================================
# Image Encoding/Decoding
# =============================================================================


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


# =============================================================================
# Observation Class
# =============================================================================


class Observation(BaseModel):
    """
    Extensible observation for any robot embodiment.

    Structure:
    - `rgb`: Camera images (keyed by camera name)
    - `depth`: Depth images (keyed by camera name)
    - `proprio`: Proprioceptive channels (keyed by channel name)
    - `cam_intrinsics`: Camera intrinsics matrices (keyed by camera name)
    - `cam_extrinsics`: Camera extrinsics matrices (keyed by camera name)
    - `extra`: Arbitrary additional data

    Examples:
        # Single-arm manipulator
        obs = Observation(
            rgb={"wrist": img},
            proprio={
                "ee_pos": [0.4, 0.0, 0.2],
                "ee_quat": [0, 0, 0, 1],
                "gripper": [0.5],
            },
        )

        # Navigation agent
        obs = Observation(
            rgb={"ego": img},
            proprio={
                "base_pos": [1.2, 0.0, 3.4],
                "base_heading": [1.57],
            },
            extra={"task_prompt": "Pick up the apple."},
        )
    """

    model_config = ConfigDict(extra="forbid")

    rgb: dict[str, dict | list] = Field(default_factory=dict)
    depth: dict[str, dict | list] = Field(default_factory=dict)
    proprio: dict[str, list[float]] = Field(default_factory=dict)
    cam_intrinsics: dict[str, list[list[float]]] = Field(default_factory=dict)
    cam_extrinsics: dict[str, list[list[float]]] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("rgb", "depth", mode="before")
    @classmethod
    def _coerce_images(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    result[k] = _encode_image(v)
                else:
                    result[k] = v
            return result
        return value

    @field_validator("proprio", mode="before")
    @classmethod
    def _coerce_proprio(cls, value: Any) -> dict[str, list[float]]:
        if value is None:
            return {}
        result = {}
        for k, v in value.items():
            if isinstance(v, np.ndarray):
                result[k] = v.tolist()
            elif isinstance(v, (int, float)):
                result[k] = [float(v)]
            else:
                result[k] = list(v)
        return result

    def get_image(self, camera: str) -> np.ndarray | None:
        """Get decoded RGB image for a camera."""
        if camera not in self.rgb:
            return None
        return _decode_image(self.rgb[camera])

    def get_depth(self, camera: str) -> np.ndarray | None:
        """Get decoded depth image for a camera."""
        if camera not in self.depth:
            return None
        return _decode_image(self.depth[camera])

    @property
    def images(self) -> dict[str, np.ndarray]:
        """Get all RGB images as numpy arrays."""
        return {name: _decode_image(data) for name, data in self.rgb.items()}

    def get_proprio(self, key: str) -> np.ndarray | None:
        """Get a proprioceptive channel as numpy array."""
        if key not in self.proprio:
            return None
        return np.array(self.proprio[key], dtype=np.float32)

    def proprio_array(self, keys: list[str] | None = None) -> np.ndarray:
        """
        Concatenate proprio channels into flat array.

        Args:
            keys: Channel names to include (default: all, sorted alphabetically)
        """
        if keys is None:
            keys = sorted(self.proprio.keys())
        arrays = [self.proprio[k] for k in keys if k in self.proprio]
        if not arrays:
            return np.array([], dtype=np.float32)
        return np.concatenate(arrays).astype(np.float32)

    def to_payload(self, include_images: bool = True) -> dict[str, Any]:
        """Serialize for network transport."""
        data = self.model_dump(mode="python")
        if not include_images:
            data["rgb"] = {}
            data["depth"] = {}
        return data


# =============================================================================
# Action Class
# =============================================================================


class Action(BaseModel):
    """
    Extensible action for any robot embodiment.

    Structure:
    - `continuous`: Continuous control channels (normalized to [-1, 1] or [0, 1])
    - `discrete`: Discrete action selections
    - `extra`: Arbitrary additional data

    Examples:
        # Manipulator action
        action = Action(continuous={
            "ee_twist": [0.1, 0, 0, 0, 0, 0],
            "gripper": [1.0],
        })

        # Navigation action
        action = Action(
            discrete={"nav": 0},  # 0=forward
            continuous={"gripper": [0.8]},
        )
    """

    model_config = ConfigDict(extra="forbid")

    continuous: dict[str, list[float]] = Field(default_factory=dict)
    discrete: dict[str, int] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("continuous", mode="before")
    @classmethod
    def _coerce_continuous(cls, value: Any) -> dict[str, list[float]]:
        if value is None:
            return {}
        result = {}
        for k, v in value.items():
            if isinstance(v, np.ndarray):
                result[k] = v.tolist()
            elif isinstance(v, (int, float)):
                result[k] = [float(v)]
            else:
                result[k] = list(v)
        return result

    def get_continuous(self, key: str) -> np.ndarray | None:
        """Get a continuous channel as numpy array."""
        if key not in self.continuous:
            return None
        return np.array(self.continuous[key], dtype=np.float32)

    def continuous_array(self, keys: list[str] | None = None) -> np.ndarray:
        """Concatenate continuous channels into flat array."""
        if keys is None:
            keys = sorted(self.continuous.keys())
        arrays = [self.continuous[k] for k in keys if k in self.continuous]
        if not arrays:
            return np.array([], dtype=np.float32)
        return np.concatenate(arrays).astype(np.float32)

    @classmethod
    def from_array(cls, array: Any, schema: dict[str, int]) -> Action:
        """
        Construct from flat array given a schema.

        Args:
            array: Flat numpy array of action values
            schema: Mapping of channel names to sizes, e.g. {"ee_twist": 6, "gripper": 1}
        """
        arr = np.asarray(array, dtype=np.float32).flatten()
        continuous: dict[str, list[float]] = {}
        idx = 0
        for key, size in schema.items():
            continuous[key] = arr[idx : idx + size].tolist()
            idx += size
        return cls(continuous=continuous)

    def to_array(self, keys: list[str] | None = None) -> np.ndarray:
        """Convert to flat array for backward compatibility."""
        return self.continuous_array(keys)


# =============================================================================
# Helper Functions
# =============================================================================


def coerce_action(action: Any, schema: dict[str, int] | None = None) -> Action:
    """
    Coerce various formats to Action.

    Args:
        action: Action in any supported format (Action, dict, or array)
        schema: For array inputs, mapping of channel names to sizes
    """
    if isinstance(action, Action):
        return action
    if isinstance(action, dict):
        return Action.model_validate(action)
    if isinstance(action, np.ndarray) or hasattr(action, "__iter__"):
        if schema is None:
            # Default schema for backward compatibility with 7-element actions
            schema = {ActionKeys.EE_TWIST: 6, ActionKeys.GRIPPER: 1}
        return Action.from_array(np.asarray(action), schema)
    raise TypeError(f"Cannot coerce {type(action)} to Action")


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    """Normalize a quaternion to unit length."""
    norm = np.linalg.norm(quat)
    if norm <= 0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quat / norm).astype(np.float32)


class Step(BaseModel):
    """A step containing observation and action (for trajectories)."""

    model_config = ConfigDict(extra="forbid")

    obs: Observation
    action: Action
