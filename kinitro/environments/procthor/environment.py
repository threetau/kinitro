"""ProcTHOR procedural environment for embodied AI tasks."""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from kinitro.environments.base import RoboticsEnvironment, TaskConfig
from kinitro.environments.procthor.house_generator import (
    HouseGenerator,
    extract_scene_objects,
)
from kinitro.environments.procthor.task_generator import TaskGenerator
from kinitro.environments.procthor.task_types import SceneObject, TaskSpec, TaskType
from kinitro.rl_interface import CanonicalAction, CanonicalObservation, coerce_action

logger = structlog.get_logger()


class ProcTHOREnvironment(RoboticsEnvironment):
    """
    ProcTHOR environment with procedural houses and scene-grounded tasks.

    This environment:
    - Generates procedural houses using ProcTHOR
    - Creates scene-grounded task prompts (e.g., "Pick up the apple")
    - Supports navigation + manipulation actions
    - Validates task feasibility before returning tasks

    Actions are mapped from CanonicalAction (twist + gripper) to AI2-THOR:
    - Linear velocity (twist[0:3]) -> MoveAgent, MoveArm
    - Angular velocity (twist[3:6]) -> RotateAgent, LookUp/Down
    - Gripper (0-1) -> PickupObject, ReleaseObject
    """

    # Thresholds for action selection
    MOVE_THRESHOLD = 0.2
    ROTATE_THRESHOLD = 0.2
    GRIPPER_PICKUP_THRESHOLD = 0.7
    GRIPPER_RELEASE_THRESHOLD = 0.3

    # Movement scales
    MOVE_SCALE = 0.25  # meters per action
    ROTATE_SCALE = 30.0  # degrees per action
    ARM_MOVE_SCALE = 0.1  # meters per arm movement

    def __init__(
        self,
        task_name: str = "procthor-v0",
        image_size: tuple[int, int] = (300, 300),
        field_of_view: int = 90,
        visibility_distance: float = 1.5,
        grid_size: float = 0.25,
        use_depth: bool = False,
        task_types: list[TaskType] | None = None,
        max_episode_steps: int = 500,
        headless: bool = True,
    ) -> None:
        """
        Initialize the ProcTHOR environment.

        Args:
            task_name: Name identifier for this environment
            image_size: (width, height) for rendered images
            field_of_view: Camera field of view in degrees
            visibility_distance: Max distance for object visibility
            grid_size: Navigation grid size in meters
            use_depth: Whether to render depth images
            task_types: Which task types to generate (None = all)
            max_episode_steps: Maximum steps per episode
            headless: Run without display window (default True for server use)
        """
        self._task_name = task_name
        self._width, self._height = image_size
        self._field_of_view = field_of_view
        self._visibility_distance = visibility_distance
        self._grid_size = grid_size
        self._use_depth = use_depth
        self._max_episode_steps = max_episode_steps
        self._headless = headless

        # Lazy-initialized components
        self._controller = None
        self._house_generator = HouseGenerator()
        self._task_generator = TaskGenerator(task_types=task_types)

        # Episode state
        self._current_task: TaskSpec | None = None
        self._current_house: dict[str, Any] | None = None
        self._scene_objects: list[SceneObject] = []
        self._episode_steps = 0
        self._episode_success = False
        self._prev_gripper_state = 0.0
        self._holding_object: str | None = None

        # For velocity estimation
        self._prev_agent_pos: np.ndarray | None = None
        self._prev_agent_rot: float | None = None

    @property
    def env_name(self) -> str:
        return "procthor"

    @property
    def task_name(self) -> str:
        return self._task_name

    def _ensure_controller(self) -> None:
        """Lazy initialization of AI2-THOR controller."""
        if self._controller is not None:
            return

        try:
            from ai2thor.controller import Controller
        except ImportError as exc:
            raise ImportError("ai2thor is required. Install with: pip install ai2thor") from exc

        # For procedural house generation, we need to use the nanna branch of AI2-THOR.
        # The nanna branch has builds available (e.g., fdb56f1c53c9) that support CreateHouse.
        # Note: We hardcode 'nanna' instead of using PROCTHOR_INITIALIZATION because newer
        # procthor versions changed to 'main' branch, which has no builds available.
        controller_kwargs = {
            "agentMode": "arm",
            "massThreshold": None,
            "visibilityDistance": self._visibility_distance,
            "gridSize": self._grid_size,
            "renderDepthImage": self._use_depth,
            "renderInstanceSegmentation": False,
            "width": self._width,
            "height": self._height,
            "fieldOfView": self._field_of_view,
            # ProcTHOR requires branch='nanna' and scene='Procedural' for house generation
            "branch": "nanna",
            "scene": "Procedural",
        }

        # For headless mode, we have two options:
        # 1. CloudRendering - requires Vulkan/GPU (faster, but needs hardware)
        # 2. Linux64 + Xvfb - works anywhere, but slower
        #
        # Set AI2THOR_PLATFORM=Linux64 to skip CloudRendering attempt
        import os

        platform_override = os.environ.get("AI2THOR_PLATFORM", "").lower()

        if self._headless and platform_override != "linux64":
            # Try CloudRendering if Vulkan appears to be available
            if self._vulkan_available():
                try:
                    from ai2thor.platform import CloudRendering

                    self._controller = Controller(platform=CloudRendering, **controller_kwargs)
                    logger.info("ai2thor_initialized", mode="CloudRendering")
                    return
                except Exception as e:
                    logger.warning(
                        "cloud_rendering_failed",
                        error=str(e),
                        msg="Falling back to Xvfb + Linux64",
                    )

        # Start Xvfb for headless X11 rendering (fallback for non-GPU environments)
        if self._headless:
            self._start_xvfb()

        # Use Linux64 platform with X server / Xvfb display
        self._controller = Controller(**controller_kwargs)
        logger.info("ai2thor_initialized", mode="Linux64")

    def _vulkan_available(self) -> bool:
        """Check if Vulkan appears to be available."""
        import subprocess

        try:
            result = subprocess.run(
                ["vulkaninfo"],
                capture_output=True,
                timeout=5,
            )
            # vulkaninfo returns 0 if Vulkan devices are found
            return result.returncode == 0 and b"GPU" in result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _start_xvfb(self) -> None:
        """Start Xvfb for headless X11 rendering if not already running."""
        import os
        import subprocess

        display = os.environ.get("DISPLAY", ":0")

        # Check if Xvfb is already running on this display
        try:
            result = subprocess.run(
                ["xdpyinfo", "-display", display],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                logger.info("xvfb_already_running", display=display)
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Start Xvfb
        display.lstrip(":")
        try:
            subprocess.Popen(
                [
                    "Xvfb",
                    display,
                    "-screen",
                    "0",
                    f"{self._width}x{self._height}x24",
                    "-ac",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("xvfb_started", display=display)
            # Give Xvfb a moment to start
            import time

            time.sleep(1)
        except FileNotFoundError:
            logger.warning("xvfb_not_found", msg="Xvfb not installed, continuing without it")

    def generate_task(self, seed: int) -> TaskConfig:
        """
        Generate a procedural task from seed.

        This:
        1. Generates a procedural house
        2. Initializes the scene to extract objects
        3. Generates a feasible task grounded in the scene
        """
        self._ensure_controller()

        # Generate house - pass our controller to avoid ProcTHOR creating its own
        # (which would trigger a separate binary download)
        house = self._house_generator.generate_house(seed, controller=self._controller)
        scene_name = self._house_generator.get_scene_name(house)

        # Reset to the scene to get object metadata
        if isinstance(scene_name, dict):
            # Procedural house
            event = self._controller.reset(scene=scene_name)
        else:
            # iTHOR fallback scene
            event = self._controller.reset(scene=scene_name)

        if event is None or not event.metadata.get("lastActionSuccess", True):
            raise RuntimeError(f"Failed to reset scene: {scene_name}")

        # Extract objects from scene
        objects = extract_scene_objects(event.metadata)

        # Generate task
        rng = np.random.default_rng(seed)
        task_spec = self._task_generator.generate_task(objects, rng)

        if task_spec is None:
            # Fallback: simple exploration task
            task_spec = TaskSpec(
                task_type=TaskType.PICKUP,
                task_prompt="Explore the environment.",
                target_object_id="",
                target_object_type="",
            )
            logger.warning("task_generation_fallback", seed=seed)

        return TaskConfig(
            env_name=self.env_name,
            task_name=self._task_name,
            seed=seed,
            domain_randomization={
                "house": house,
                "task_spec": task_spec.to_dict(),
            },
        )

    def reset(self, task_config: TaskConfig) -> CanonicalObservation:
        """Reset environment with the given task configuration."""
        self._ensure_controller()

        # Reset episode state
        self._episode_steps = 0
        self._episode_success = False
        self._prev_gripper_state = 0.0
        self._holding_object = None
        self._prev_agent_pos = None
        self._prev_agent_rot = None

        # Extract house and task from config
        house = task_config.domain_randomization.get("house", {})
        task_dict = task_config.domain_randomization.get("task_spec", {})

        self._current_house = house
        self._current_task = TaskSpec.from_dict(task_dict) if task_dict else None

        # Reset scene
        scene_name = self._house_generator.get_scene_name(house)
        if isinstance(scene_name, dict):
            event = self._controller.reset(scene=scene_name)
        else:
            event = self._controller.reset(scene=scene_name)

        if event is None or not event.metadata.get("lastActionSuccess", True):
            raise RuntimeError("Failed to reset ProcTHOR scene")

        # Extract scene objects
        self._scene_objects = extract_scene_objects(event.metadata)

        # Randomize agent start position
        rng = np.random.default_rng(task_config.seed)
        self._randomize_agent_position(rng)

        # Get initial observation
        event = self._controller.step(action="Pass")
        return self._build_observation(event)

    def _randomize_agent_position(self, rng: np.random.Generator) -> None:
        """Teleport agent to spawn position from house metadata.

        For procedural houses, we use the spawn position from house metadata
        since GetReachablePositions may not work reliably.
        """
        # Get spawn position from house metadata
        house_metadata = self._current_house.get("metadata", {})
        agent_spawn = house_metadata.get("agent", {})

        if agent_spawn.get("position"):
            # Use position from house metadata
            pos = agent_spawn["position"]
            rotation = agent_spawn.get("rotation", {"x": 0, "y": 0, "z": 0})
            horizon = agent_spawn.get("horizon", 30)

            # Optionally randomize rotation
            if rng is not None:
                rotation = {"x": 0.0, "y": float(rng.choice([0, 90, 180, 270])), "z": 0.0}

            self._controller.step(
                action="Teleport",
                position=pos,
                rotation=rotation,
                horizon=float(horizon),
                standing=True,
            )
        else:
            # Fallback: try GetReachablePositions
            event = self._controller.step(action="GetReachablePositions")
            positions = event.metadata.get("actionReturn", [])

            if positions:
                pos = positions[rng.integers(0, len(positions))]
                rotation = float(rng.choice([0, 90, 180, 270]))

                self._controller.step(
                    action="Teleport",
                    position=pos,
                    rotation={"x": 0.0, "y": rotation, "z": 0.0},
                    horizon=30.0,
                    standing=True,
                )

    def step(
        self, action: CanonicalAction | dict[str, Any] | np.ndarray
    ) -> tuple[CanonicalObservation, float, bool, dict[str, Any]]:
        """Execute action in environment."""
        self._ensure_controller()
        self._episode_steps += 1

        canonical_action = coerce_action(action)
        twist = np.clip(np.array(canonical_action.twist_ee_norm, dtype=np.float32), -1.0, 1.0)
        gripper = float(np.clip(canonical_action.gripper_01, 0.0, 1.0))

        # Execute action based on twist and gripper
        event, action_name = self._execute_action(twist, gripper)

        if event is None:
            event = self._controller.step(action="Pass")
            action_name = "Pass"

        # Update scene objects
        self._scene_objects = extract_scene_objects(event.metadata)

        # Check for task completion
        self._update_success()

        # Build observation
        obs = self._build_observation(event)

        # Compute reward (sparse: 1.0 on success, 0.0 otherwise)
        reward = 1.0 if self._episode_success else 0.0

        # Check termination
        done = self._episode_success or self._episode_steps >= self._max_episode_steps

        info = {
            "task_prompt": self._current_task.task_prompt if self._current_task else "",
            "task_type": (self._current_task.task_type.value if self._current_task else ""),
            "episode_steps": self._episode_steps,
            "success": self._episode_success,
            "last_action": action_name,
            "last_action_success": event.metadata.get("lastActionSuccess", False),
            "error_message": event.metadata.get("errorMessage", ""),
        }

        return obs, reward, done, info

    def _execute_action(self, twist: np.ndarray, gripper: float) -> tuple[Any, str]:
        """
        Map canonical action to AI2-THOR actions.

        Priority:
        1. Gripper actions (pickup/release)
        2. Navigation (body movement/rotation)
        3. Arm movement

        Returns:
            Tuple of (event, action_name)
        """
        # Handle gripper state changes
        if gripper > self.GRIPPER_PICKUP_THRESHOLD and self._holding_object is None:
            event, action_name = self._try_pickup()
            if event is not None:
                self._prev_gripper_state = gripper
                return event, action_name

        if gripper < self.GRIPPER_RELEASE_THRESHOLD and self._holding_object is not None:
            event, action_name = self._try_release()
            if event is not None:
                self._prev_gripper_state = gripper
                return event, action_name

        # Navigation: rotation (yaw)
        if abs(twist[5]) > self.ROTATE_THRESHOLD:  # yaw rotation
            degrees = self.ROTATE_SCALE if twist[5] > 0 else -self.ROTATE_SCALE
            event = self._controller.step(
                action="RotateAgent",
                degrees=degrees,
                returnToStart=True,
                speed=1,
                fixedDeltaTime=0.02,
            )
            return event, "RotateAgent"

        # Navigation: look up/down (pitch)
        if abs(twist[3]) > self.ROTATE_THRESHOLD:  # pitch rotation
            if twist[3] > 0:
                event = self._controller.step(action="LookDown")
                return event, "LookDown"
            else:
                event = self._controller.step(action="LookUp")
                return event, "LookUp"

        # Navigation: body movement
        forward = twist[2] * self.MOVE_SCALE
        right = twist[0] * self.MOVE_SCALE

        if abs(forward) > 0.01 or abs(right) > 0.01:
            event = self._controller.step(
                action="MoveAgent",
                ahead=float(forward),
                right=float(right),
                returnToStart=True,
                speed=1,
                fixedDeltaTime=0.02,
            )
            return event, "MoveAgent"

        # Arm movement (vertical)
        if abs(twist[1]) > self.MOVE_THRESHOLD:
            arm_y = 0.5 + twist[1] * 0.3  # Map to [0.2, 0.8] range
            arm_y = float(np.clip(arm_y, 0.0, 1.0))
            event = self._controller.step(
                action="MoveArmBase",
                y=arm_y,
                speed=1,
                returnToStart=True,
                fixedDeltaTime=0.02,
            )
            return event, "MoveArmBase"

        # No significant action - pass
        return None, "Pass"

    def _try_pickup(self) -> tuple[Any, str]:
        """Attempt to pick up an object."""
        # Get objects the arm can pick up
        event = self._controller.last_event
        arm_data = event.metadata.get("arm", {})
        pickupable = arm_data.get("pickupableObjects", [])

        if pickupable:
            event = self._controller.step(
                action="PickupObject",
                objectIdCandidates=pickupable,
            )
            if event.metadata.get("lastActionSuccess", False):
                # Track what we picked up
                inventory = event.metadata.get("inventoryObjects", [])
                if inventory:
                    self._holding_object = inventory[0].get("objectId")
            return event, "PickupObject"

        return None, "PickupObject"

    def _try_release(self) -> tuple[Any, str]:
        """Attempt to release held object."""
        if self._holding_object is None:
            return None, "ReleaseObject"

        event = self._controller.step(action="ReleaseObject")
        if event.metadata.get("lastActionSuccess", False):
            self._holding_object = None
        return event, "ReleaseObject"

    def _update_success(self) -> None:
        """Check if the current task has been completed."""
        if self._current_task is None:
            self._episode_success = False
            return

        task = self._current_task
        target_id = task.target_object_id

        # Find the target object in current scene state
        target_obj = None
        for obj in self._scene_objects:
            if obj.object_id == target_id:
                target_obj = obj
                break

        if target_obj is None and target_id:
            # Object not found - might have been picked up
            self._episode_success = False
            return

        # Check completion based on task type
        if task.task_type == TaskType.PICKUP:
            # Success if object is in inventory
            self._episode_success = self._holding_object == target_id

        elif task.task_type == TaskType.PLACE:
            # Success if object is on destination receptacle
            if target_obj is not None and task.destination_object_id:
                self._episode_success = task.destination_object_id in target_obj.parent_receptacles

        elif task.task_type == TaskType.OPEN:
            # Success if object is now open
            if target_obj is not None:
                self._episode_success = target_obj.is_open

        elif task.task_type == TaskType.CLOSE:
            # Success if object is now closed
            if target_obj is not None:
                self._episode_success = not target_obj.is_open

        elif task.task_type == TaskType.TOGGLE_ON:
            # Success if object is now toggled on
            if target_obj is not None:
                self._episode_success = target_obj.is_toggled

        elif task.task_type == TaskType.TOGGLE_OFF:
            # Success if object is now toggled off
            if target_obj is not None:
                self._episode_success = not target_obj.is_toggled

    def _build_observation(self, event: Any) -> CanonicalObservation:
        """Build canonical observation from AI2-THOR event."""
        agent = event.metadata.get("agent", {})
        position = agent.get("position", {})
        rotation = agent.get("rotation", {})

        # Agent position
        pos = np.array(
            [
                position.get("x", 0.0),
                position.get("y", 0.0),
                position.get("z", 0.0),
            ],
            dtype=np.float32,
        )

        # Agent rotation (yaw only) to quaternion
        yaw_deg = float(rotation.get("y", 0.0))
        quat = self._yaw_to_quaternion(yaw_deg)

        # Estimate velocities
        if self._prev_agent_pos is None:
            lin_vel = np.zeros(3, dtype=np.float32)
            ang_vel = np.zeros(3, dtype=np.float32)
        else:
            lin_vel = pos - self._prev_agent_pos
            yaw_diff = yaw_deg - (self._prev_agent_rot or 0.0)
            ang_vel = np.array([0.0, np.deg2rad(yaw_diff), 0.0], dtype=np.float32)

        self._prev_agent_pos = pos.copy()
        self._prev_agent_rot = yaw_deg

        # Gripper state (0 = empty, 1 = holding)
        gripper_01 = 1.0 if self._holding_object else 0.0

        # RGB image
        rgb = {}
        if event.frame is not None:
            rgb["ego"] = event.frame.tolist()

        # Depth (optional)
        depth = None
        if self._use_depth and hasattr(event, "depth_frame") and event.depth_frame is not None:
            depth = event.depth_frame.tolist()

        return CanonicalObservation(
            ee_pos_m=pos.tolist(),
            ee_quat_xyzw=quat.tolist(),
            ee_lin_vel_mps=lin_vel.tolist(),
            ee_ang_vel_rps=ang_vel.tolist(),
            gripper_01=gripper_01,
            rgb=rgb,
            depth=depth,
            cam_intrinsics_K=None,
            cam_extrinsics_T_world_cam=None,
        )

    def _yaw_to_quaternion(self, yaw_deg: float) -> np.ndarray:
        """Convert yaw angle (degrees) to quaternion (xyzw)."""
        yaw = np.deg2rad(yaw_deg)
        half = 0.5 * yaw
        return np.array([0.0, np.sin(half), 0.0, np.cos(half)], dtype=np.float32)

    def get_success(self) -> bool:
        """Check if task was completed successfully."""
        return self._episode_success

    def close(self) -> None:
        """Clean up environment resources."""
        if self._controller is not None:
            try:
                self._controller.stop()
            except Exception as e:
                logger.warning("controller_close_failed", error=str(e))
            self._controller = None

    def render(self) -> np.ndarray | None:
        """Render current frame."""
        if self._controller is None:
            return None
        event = self._controller.last_event
        if event is not None and event.frame is not None:
            return event.frame
        return None

    def get_task_prompt(self) -> str:
        """Get the current task prompt."""
        if self._current_task is None:
            return ""
        return self._current_task.task_prompt
