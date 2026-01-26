"""AI2-THOR ManipulaTHOR environment wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from kinitro.environments.base import RoboticsEnvironment, TaskConfig
from kinitro.rl_interface import CanonicalAction, CanonicalObservation, coerce_action

logger = structlog.get_logger()


class AI2ThorManipulationEnvironment(RoboticsEnvironment):
    """Manipulation environment using AI2-THOR's arm agent."""

    def __init__(
        self,
        task_name: str = "manip-v0",
        scene: str = "FloorPlan203",
        image_size: tuple[int, int] = (300, 300),
        field_of_view: int = 60,
        visibility_distance: float = 1.5,
        grid_size: float = 0.25,
        use_depth: bool = False,
        move_scale: float = 0.2,
        rotate_degrees: float = 30.0,
        move_threshold: float = 0.2,
        rotate_threshold: float = 0.2,
        action_repeat: int = 1,
    ) -> None:
        self._task_name = task_name
        self._scene = scene
        self._width, self._height = image_size
        self._field_of_view = field_of_view
        self._visibility_distance = visibility_distance
        self._grid_size = grid_size
        self._use_depth = use_depth
        self._move_scale = move_scale
        self._rotate_degrees = rotate_degrees
        self._move_threshold = move_threshold
        self._rotate_threshold = rotate_threshold
        self._action_repeat = max(1, int(action_repeat))

        self._controller = None
        self._prev_pos: np.ndarray | None = None
        self._prev_yaw: float | None = None
        self._episode_success = False
        self._target_object_id: str | None = None
        self._target_object_type: str | None = None
        self._target_position: np.ndarray | None = None
        self._warned_gripper = False
        self._camera_horizon: float = 0.0
        self._rotation_cooldown = 0
        self._arm_home = np.array([0.0, 0.0, 0.5], dtype=np.float32)

    @property
    def env_name(self) -> str:
        return "ai2thor"

    @property
    def task_name(self) -> str:
        return self._task_name

    def _ensure_controller(self) -> None:
        if self._controller is not None:
            return

        try:
            from ai2thor.controller import Controller
        except ImportError as exc:
            raise ImportError("ai2thor is required. Install with: pip install ai2thor") from exc

        self._controller = Controller(
            agentMode="arm",
            massThreshold=None,
            scene=self._scene,
            visibilityDistance=self._visibility_distance,
            gridSize=self._grid_size,
            renderDepthImage=self._use_depth,
            renderInstanceSegmentation=False,
            width=self._width,
            height=self._height,
            fieldOfView=self._field_of_view,
        )

    def generate_task(self, seed: int) -> TaskConfig:
        return TaskConfig(
            env_name=self.env_name,
            task_name=self._task_name,
            seed=seed,
            domain_randomization={
                "scene": self._scene,
            },
        )

    def _agent_pose(self, event: Any) -> tuple[np.ndarray, float]:
        agent = event.metadata.get("agent", {})
        arm = agent.get("arm", {})
        hand = arm.get("hand", {})
        position = hand.get("position") or agent.get("position", {})
        rotation = agent.get("rotation", {})

        pos = np.array(
            [position.get("x", 0.0), position.get("y", 0.0), position.get("z", 0.0)],
            dtype=np.float32,
        )
        yaw = float(rotation.get("y", 0.0))
        return pos, yaw

    def _yaw_to_quaternion(self, yaw_deg: float) -> np.ndarray:
        yaw = np.deg2rad(yaw_deg)
        half = 0.5 * yaw
        return np.array([0.0, np.sin(half), 0.0, np.cos(half)], dtype=np.float32)

    def _build_observation(self, event: Any) -> CanonicalObservation:
        self._update_camera_horizon(event)
        pos, yaw = self._agent_pose(event)
        quat = self._yaw_to_quaternion(yaw)

        if self._prev_pos is None:
            lin_vel = np.zeros(3, dtype=np.float32)
        else:
            lin_vel = (pos - self._prev_pos) / self._grid_size

        if self._prev_yaw is None:
            ang_vel = np.zeros(3, dtype=np.float32)
        else:
            yaw_rate = np.deg2rad(yaw - self._prev_yaw) / self._grid_size
            ang_vel = np.array([0.0, yaw_rate, 0.0], dtype=np.float32)

        self._prev_pos = pos
        self._prev_yaw = yaw

        rgb = {}
        if event.frame is not None:
            rgb = {"ego": event.frame.tolist()}

        depth = None
        if self._use_depth and getattr(event, "depth_frame", None) is not None:
            depth = event.depth_frame.tolist()

        return CanonicalObservation(
            ee_pos_m=pos.tolist(),
            ee_quat_xyzw=quat.tolist(),
            ee_lin_vel_mps=lin_vel.tolist(),
            ee_ang_vel_rps=ang_vel.tolist(),
            gripper_01=0.0,
            rgb=rgb,
            depth=depth,
            cam_intrinsics_K=None,
            cam_extrinsics_T_world_cam=None,
        )

    def _select_action(self, twist: np.ndarray) -> tuple[str, dict[str, Any]]:
        move = twist[:3]
        rot = twist[3:6]

        if self._rotation_cooldown == 0 and abs(rot[1]) > self._rotate_threshold:
            degrees = self._rotate_degrees if rot[1] > 0 else -self._rotate_degrees
            return "RotateAgent", {
                "degrees": degrees,
                "returnToStart": True,
            }

        if rot[0] > self._rotate_threshold and self._camera_horizon < 60.0:
            return "LookDown", {}
        if rot[0] < -self._rotate_threshold and self._camera_horizon > -30.0:
            return "LookUp", {}

        if abs(move[1]) > self._move_threshold:
            return "MoveArm", {
                "position": {
                    "x": 0.0,
                    "y": float(move[1] * self._move_scale),
                    "z": float(self._arm_home[2]),
                },
                "coordinateSpace": "armBase",
                "restrictMovement": False,
                "returnToStart": True,
                "speed": 1,
                "fixedDeltaTime": 0.02,
            }

        if abs(move[2]) > self._move_threshold or abs(move[0]) > self._move_threshold:
            return "MoveAgent", {
                "ahead": float(move[2] * self._move_scale),
                "right": float(move[0] * self._move_scale),
                "returnToStart": True,
                "speed": 1,
                "fixedDeltaTime": 0.02,
            }

        return "Pass", {}

    def _update_camera_horizon(self, event: Any) -> None:
        agent = event.metadata.get("agent", {})
        horizon = agent.get("cameraHorizon")
        if horizon is not None:
            self._camera_horizon = float(horizon)

    def _reset_arm_pose(self) -> None:
        if self._controller is None:
            return
        self._controller.step(
            action="MoveArm",
            position={"x": 0.0, "y": 0.0, "z": float(self._arm_home[2])},
            coordinateSpace="armBase",
            restrictMovement=False,
            returnToStart=True,
            speed=1,
            fixedDeltaTime=0.02,
        )

    def _pick_target_object(self, event: Any, rng: np.random.Generator) -> None:
        objects = [obj for obj in event.metadata.get("objects", []) if obj.get("pickupable")]
        if not objects:
            self._target_object_id = None
            self._target_object_type = None
            self._target_position = None
            return
        choice = rng.choice(objects)
        self._target_object_id = choice.get("objectId")
        self._target_object_type = choice.get("objectType")
        position = choice.get("position", {})
        self._target_position = np.array(
            [position.get("x", 0.0), position.get("y", 0.0), position.get("z", 0.0)],
            dtype=np.float32,
        )

    def _update_success(self, event: Any) -> None:
        if not self._target_object_id:
            self._episode_success = False
            return
        inventory = event.metadata.get("inventoryObjects", [])
        self._episode_success = any(
            obj.get("objectId") == self._target_object_id for obj in inventory
        )

    def _choose_start_pose(
        self, reachable: list[dict[str, float]] | None
    ) -> tuple[dict[str, float] | None, float]:
        if not reachable:
            return None, 0.0
        if self._target_position is None:
            pos = reachable[0]
            return pos, 0.0
        best = None
        best_dist = float("inf")
        for pos in reachable:
            dx = self._target_position[0] - pos.get("x", 0.0)
            dz = self._target_position[2] - pos.get("z", 0.0)
            dist = dx * dx + dz * dz
            if dist < best_dist:
                best_dist = dist
                best = pos
        if best is None:
            return reachable[0], 0.0
        dx = self._target_position[0] - best.get("x", 0.0)
        dz = self._target_position[2] - best.get("z", 0.0)
        yaw = float(np.degrees(np.arctan2(dx, dz))) if dx or dz else 0.0
        return best, yaw

    def _maybe_handle_gripper(self, event: Any, gripper: float) -> Any:
        if self._controller is None:
            return event
        if gripper > 0.5:
            candidates = event.metadata.get("arm", {}).get("pickupableObjects", [])
            if candidates:
                event = self._controller.step(
                    action="PickupObject",
                    objectIdCandidates=candidates,
                )
        elif gripper < 0.1:
            inventory = event.metadata.get("inventoryObjects", [])
            if inventory:
                event = self._controller.step(action="ReleaseObject")
        return event

    def reset(self, task_config: TaskConfig) -> CanonicalObservation:
        self._ensure_controller()
        self._episode_success = False
        self._prev_pos = None
        self._prev_yaw = None
        self._warned_gripper = False
        self._rotation_cooldown = 0
        self._camera_horizon = 0.0

        event = self._controller.reset(
            scene=self._scene,
            agentMode="arm",
            massThreshold=None,
            visibilityDistance=self._visibility_distance,
            gridSize=self._grid_size,
            renderDepthImage=self._use_depth,
            renderInstanceSegmentation=False,
            width=self._width,
            height=self._height,
            fieldOfView=self._field_of_view,
        )

        if event is None or not event.metadata.get("lastActionSuccess", True):
            raise RuntimeError("Failed to reset AI2-THOR scene.")

        rng = np.random.default_rng(task_config.seed)
        reachable_event = self._controller.step(action="GetReachablePositions")
        reachable = reachable_event.metadata.get("actionReturn") if reachable_event else None

        self._pick_target_object(event, rng)
        start_pos, yaw = self._choose_start_pose(reachable)
        if start_pos:
            event = self._controller.step(
                action="Teleport",
                position=start_pos,
                rotation={"x": 0.0, "y": yaw, "z": 0.0},
                horizon=45.0,
                standing=True,
            )
            if event is not None:
                self._camera_horizon = float(
                    event.metadata.get("agent", {}).get("cameraHorizon", 45.0)
                )

        if self._camera_horizon < 60.0:
            event = self._controller.step(action="LookDown")
            if event is not None:
                self._camera_horizon = float(
                    event.metadata.get("agent", {}).get("cameraHorizon", self._camera_horizon)
                )

        self._reset_arm_pose()

        self._update_success(event)
        return self._build_observation(event)

    def step(
        self, action: CanonicalAction | dict[str, Any] | np.ndarray
    ) -> tuple[CanonicalObservation, float, bool, dict[str, Any]]:
        canonical_action = coerce_action(action)
        twist = np.clip(np.array(canonical_action.twist_ee_norm, dtype=np.float32), -1.0, 1.0)

        if self._rotation_cooldown > 0:
            self._rotation_cooldown -= 1

        action_name, payload = self._select_action(twist)
        event = None
        for _ in range(self._action_repeat):
            if action_name in {"RotateAgent", "MoveAgent"}:
                self._reset_arm_pose()
            event = self._controller.step(action=action_name, **payload)

        if event is None:
            raise RuntimeError("AI2-THOR step returned no event.")

        if not event.metadata.get("lastActionSuccess", True):
            logger.debug(
                "ai2thor_action_failed",
                action=action_name,
                error_message=event.metadata.get("errorMessage"),
            )
            if action_name == "RotateAgent":
                self._rotation_cooldown = 3

        event = self._maybe_handle_gripper(event, canonical_action.gripper_01)
        obs = self._build_observation(event)
        self._update_success(event)
        reward = 1.0 if self._episode_success else 0.0
        done = self._episode_success
        info = {
            "last_action": action_name,
            "last_action_success": event.metadata.get("lastActionSuccess"),
            "error_message": event.metadata.get("errorMessage"),
            "camera_horizon": self._camera_horizon,
            "target_object_id": self._target_object_id,
            "target_object_type": self._target_object_type,
            "pickupable_objects": event.metadata.get("arm", {}).get("pickupableObjects", []),
            "inventory_objects": [
                obj.get("objectId") for obj in event.metadata.get("inventoryObjects", [])
            ],
        }
        return obs, reward, done, info

    def get_success(self) -> bool:
        return self._episode_success

    def close(self) -> None:
        if self._controller is not None:
            try:
                self._controller.stop()
            except Exception as exc:
                logger.warning("ai2thor_controller_close_failed", error=str(exc))
            self._controller = None
