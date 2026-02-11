"""Base class for all Genesis simulation environments."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from abc import abstractmethod
from typing import Any

import numpy as np
import structlog

from kinitro.environments.base import RoboticsEnvironment, TaskConfig
from kinitro.environments.genesis.robot_config import RobotConfig
from kinitro.environments.genesis.scene_generator import (
    SceneConfig,
    SceneGenerator,
    SceneObjectConfig,
)
from kinitro.environments.genesis.task_generator import TaskGenerator
from kinitro.environments.genesis.task_types import SceneObject, TaskSpec, TaskType
from kinitro.rl_interface import Action, ActionKeys, Observation, ProprioKeys, encode_image

logger = structlog.get_logger()

# Track whether Genesis has been initialized (global, once per process)
_genesis_initialized = False


def _detect_render_platform() -> None:
    """Auto-detect the best OpenGL platform for Genesis rendering.

    On Linux, probes EGL availability (requires GPU + EGL libraries).
    Falls back to OSMesa (CPU software rendering) when EGL is unavailable.
    Respects an existing PYOPENGL_PLATFORM env var without overriding it.
    On non-Linux platforms, does nothing (Genesis uses pyglet by default).
    """
    if os.environ.get("PYOPENGL_PLATFORM"):
        logger.info(
            "render_platform_preset",
            platform=os.environ["PYOPENGL_PLATFORM"],
        )
        return

    if sys.platform != "linux":
        return

    # Probe EGL via ctypes (not PyOpenGL) to avoid locking PyOpenGL's
    # platform before we know whether EGL actually works.
    #
    # In headless Docker containers with NVIDIA GPUs, the default EGL
    # display (EGL_DEFAULT_DISPLAY) often has zero renderable configs
    # because there is no X11/Wayland display server.  The correct
    # approach is to enumerate EGL devices via eglQueryDevicesEXT and
    # open a platform display for the first GPU device.  We fall back
    # to the legacy eglGetDisplay path when the extension is absent.
    try:
        egl = ctypes.CDLL(ctypes.util.find_library("EGL") or "libEGL.so.1")
        display = _egl_get_device_display(egl) or _egl_get_default_display(egl)
        if not display:
            raise RuntimeError("no usable EGL display found")

        major, minor = ctypes.c_int(), ctypes.c_int()
        egl.eglInitialize.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        egl.eglInitialize.restype = ctypes.c_int
        if not egl.eglInitialize(display, ctypes.byref(major), ctypes.byref(minor)):
            raise RuntimeError("eglInitialize failed")

        # Verify renderable configs exist.  We must explicitly request
        # EGL_SURFACE_TYPE=EGL_PBUFFER_BIT because eglChooseConfig defaults
        # to EGL_WINDOW_BIT, and headless GPU configs only support pbuffers.
        # EGL constants:
        # EGL_SURFACE_TYPE=0x3033, EGL_PBUFFER_BIT=0x0001
        # EGL_RENDERABLE_TYPE=0x3040, EGL_OPENGL_BIT=0x0008, EGL_NONE=0x3038
        attribs = (ctypes.c_int * 5)(0x3033, 0x0001, 0x3040, 0x0008, 0x3038)
        config = ctypes.c_void_p()
        num_configs = ctypes.c_int()
        egl.eglChooseConfig.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        egl.eglChooseConfig.restype = ctypes.c_int
        if (
            not egl.eglChooseConfig(
                display, attribs, ctypes.byref(config), 1, ctypes.byref(num_configs)
            )
            or num_configs.value < 1
        ):
            raise RuntimeError("eglChooseConfig found no renderable configs (no GPU?)")

        egl.eglTerminate.argtypes = [ctypes.c_void_p]
        egl.eglTerminate.restype = ctypes.c_int
        egl.eglTerminate(display)

        os.environ["PYOPENGL_PLATFORM"] = "egl"
        logger.info("render_platform_detected", platform="egl")
    except Exception as exc:
        os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        logger.info(
            "render_platform_detected",
            platform="osmesa",
            reason=str(exc),
        )


def _egl_get_device_display(egl: ctypes.CDLL) -> ctypes.c_void_p | None:
    """Try to open an EGL display via device enumeration (headless-friendly).

    Uses EGL_EXT_device_enumeration + EGL_EXT_platform_device to get a
    display backed by a specific GPU, which works without a display server.
    Returns None if the extensions are unavailable.
    """
    try:
        egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
        egl.eglGetProcAddress.restype = ctypes.c_void_p

        query_ptr = egl.eglGetProcAddress(b"eglQueryDevicesEXT")
        platform_ptr = egl.eglGetProcAddress(b"eglGetPlatformDisplayEXT")
        if not query_ptr or not platform_ptr:
            return None

        FUNCTYPE_QD = ctypes.CFUNCTYPE(  # noqa: N806
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
        )
        FUNCTYPE_GP = ctypes.CFUNCTYPE(  # noqa: N806
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        )
        egl_query_devices = FUNCTYPE_QD(query_ptr)
        egl_get_platform_display = FUNCTYPE_GP(platform_ptr)

        max_devices = 8
        devices = (ctypes.c_void_p * max_devices)()
        num_devices = ctypes.c_int()
        if not egl_query_devices(max_devices, devices, ctypes.byref(num_devices)):
            return None

        # EGL_PLATFORM_DEVICE_EXT = 0x313F
        for i in range(num_devices.value):
            display = egl_get_platform_display(0x313F, devices[i], None)
            if display:
                return ctypes.c_void_p(display)
    except Exception:
        pass
    return None


def _egl_get_default_display(egl: ctypes.CDLL) -> ctypes.c_void_p | None:
    """Legacy fallback: open the EGL default display."""
    try:
        egl.eglGetDisplay.argtypes = [ctypes.c_void_p]
        egl.eglGetDisplay.restype = ctypes.c_void_p
        display = egl.eglGetDisplay(ctypes.c_void_p(0))
        return ctypes.c_void_p(display) if display else None
    except Exception:
        return None


def _patch_osmesa_framebuffers() -> None:
    """Patch Genesis's pyrender OSMesa platform to report framebuffer support.

    Genesis hardcodes ``OSMesaPlatform.supports_framebuffers = False``, which
    sends rendering down a broken code path (the non-FBO path calls
    ``renderer.render()`` without ``RenderFlags.OFFSCREEN``, so
    ``_forward_pass()`` never reads the framebuffer and returns None, hitting
    ``assert result is not None``).

    Modern OSMesa with GL 3.3 core profile (provided by ``libosmesa6-dev``)
    fully supports FBOs, so we patch the method to return True.
    """
    import importlib  # noqa: PLC0415

    mod = importlib.import_module("genesis.ext.pyrender.platforms.osmesa")
    setattr(mod.OSMesaPlatform, "supports_framebuffers", lambda self: True)
    logger.info("osmesa_framebuffer_patch_applied")


def _init_genesis() -> None:
    """Initialize Genesis engine (idempotent, once per process)."""
    global _genesis_initialized  # noqa: PLW0603
    if _genesis_initialized:
        return

    _detect_render_platform()

    # Deferred import: Genesis locks the PyOpenGL platform backend on import,
    # so _detect_render_platform() must run first (see PYOPENGL_PLATFORM notes).
    import genesis as gs  # noqa: PLC0415

    # If using OSMesa, patch FBO support before any rendering occurs.
    if os.environ.get("PYOPENGL_PLATFORM") == "osmesa":
        _patch_osmesa_framebuffers()

    try:
        gs.init(backend=getattr(gs, "gpu"))
        logger.info("genesis_initialized", backend="gpu")
    except RuntimeError as exc:
        logger.info("genesis_gpu_unavailable", error=str(exc))
        gs.init(backend=getattr(gs, "cpu"))
        logger.info("genesis_initialized", backend="cpu")

    _genesis_initialized = True


class GenesisBaseEnvironment(RoboticsEnvironment):
    """Base class for all Genesis environments.

    Handles all Genesis boilerplate:
    - Engine initialization (gpu/cpu fallback)
    - Scene creation with headless rendering
    - Robot loading from RobotConfig
    - Camera setup + image capture + tensor-to-numpy
    - PD control pipeline (action -> joint targets -> step -> read state)
    - Observation/action space definition from RobotConfig

    Subclasses implement only:
    - _get_scene_generator() -> SceneGenerator
    - _get_task_generator() -> TaskGenerator
    - _compute_reward(robot_state, object_states, task_spec) -> float
    - _check_success(robot_state, object_states, task_spec) -> bool
    """

    # Rendering settings
    IMAGE_SIZE = 84
    SIM_DT = 0.01  # 100 Hz physics
    CONTROL_DT = 0.02  # 50 Hz control (2 physics steps per control step)

    def __init__(
        self,
        robot_config: RobotConfig,
        task_name: str,
        max_episode_steps: int = 500,
        show_viewer: bool = False,
    ) -> None:
        self._robot_config = robot_config
        self._task_name = task_name
        self._max_episode_steps = max_episode_steps
        self._show_viewer = show_viewer

        # Lazy-initialized Genesis components — typed as Any because genesis
        # is an optional runtime dependency that the type checker cannot resolve.
        self._scene: Any = None
        self._robot: Any = None
        self._camera: Any = None
        self._object_entities: list[Any] = []

        # Camera availability flag — set to False if validation render fails
        self._camera_available: bool = True

        # Pre-computed arrays for action pipeline (avoid per-step allocation)
        self._default_dof_pos = np.array(robot_config.default_dof_pos, dtype=np.float32)
        self._action_scale = np.array(robot_config.action_scale, dtype=np.float32)

        # Generators (created by subclass)
        self._scene_generator: SceneGenerator | None = None
        self._task_generator: TaskGenerator | None = None

        # Episode state
        self._current_task: TaskSpec | None = None
        self._current_scene_config: SceneConfig | None = None
        self._scene_objects: list[SceneObject] = []
        self._episode_steps = 0
        self._episode_success = False

    @property
    def env_name(self) -> str:
        return "genesis"

    @property
    def task_name(self) -> str:
        return self._task_name

    @property
    def observation_shape(self) -> tuple[int, ...]:
        # proprio: base_pos(3) + base_quat(4) + base_vel(6) + joint_pos(N) + joint_vel(N)
        n = self._robot_config.num_actuated_dofs
        return (13 + 2 * n,)

    @property
    def action_shape(self) -> tuple[int, ...]:
        return (self._robot_config.num_actuated_dofs,)

    @abstractmethod
    def _get_scene_generator(self) -> SceneGenerator:
        """Return configured scene generator for this environment variant."""

    @abstractmethod
    def _get_task_generator(self) -> TaskGenerator:
        """Return configured task generator for this environment variant."""

    @abstractmethod
    def _compute_reward(
        self,
        robot_state: dict[str, np.ndarray],
        object_states: dict[str, np.ndarray],
        task_spec: TaskSpec,
    ) -> float:
        """Compute environment-specific reward."""

    @abstractmethod
    def _check_success(
        self,
        robot_state: dict[str, np.ndarray],
        object_states: dict[str, np.ndarray],
        task_spec: TaskSpec,
    ) -> bool:
        """Check environment-specific success condition."""

    def generate_task(self, seed: int) -> TaskConfig:
        """Generate a procedural task from seed."""
        if self._scene_generator is None:
            self._scene_generator = self._get_scene_generator()
        if self._task_generator is None:
            self._task_generator = self._get_task_generator()

        # Generate scene
        scene_config = self._scene_generator.generate_scene(seed)
        scene_objects = scene_config.get_scene_objects()

        # Generate task grounded in the scene
        rng = np.random.default_rng(seed)
        task_spec = self._task_generator.generate_task(
            scene_objects, rng, robot_config=self._robot_config
        )

        if task_spec is None:
            # Fallback: navigate to first object
            if scene_objects:
                target = scene_objects[0]
                task_spec = TaskSpec(
                    task_type=TaskType.NAVIGATE,
                    task_prompt=f"Walk to the {target.color} {target.object_type}.",
                    target_object_id=target.object_id,
                    target_object_type=target.object_type,
                    target_position=target.position,
                )
            else:
                task_spec = TaskSpec(
                    task_type=TaskType.NAVIGATE,
                    task_prompt="Explore the environment.",
                    target_object_id="",
                    target_object_type="",
                    target_position=[1.0, 0.0, 0.0],
                )
            logger.warning("task_generation_fallback", seed=seed)

        return TaskConfig(
            env_name=self.env_name,
            task_name=self._task_name,
            seed=seed,
            domain_randomization={
                "scene_config": {
                    "terrain_type": scene_config.terrain_type,
                    "terrain_params": scene_config.terrain_params,
                    "objects": [
                        {
                            "object_id": obj.object_id,
                            "object_type": obj.object_type,
                            "position": obj.position,
                            "color": obj.color,
                            "color_rgb": list(obj.color_rgb),
                            "size": obj.size,
                            "pickupable": obj.pickupable,
                        }
                        for obj in scene_config.objects
                    ],
                },
                "task_spec": task_spec.to_dict(),
            },
        )

    def reset(self, task_config: TaskConfig) -> Observation:
        """Reset environment with the given task configuration."""
        _init_genesis()

        # Reset episode state
        self._episode_steps = 0
        self._episode_success = False

        # Extract scene config and task spec from TaskConfig
        scene_data = task_config.domain_randomization.get("scene_config", {})
        task_dict = task_config.domain_randomization.get("task_spec", {})

        self._current_task = TaskSpec.from_dict(task_dict) if task_dict else None

        obj_configs = [
            SceneObjectConfig(
                object_id=obj["object_id"],
                object_type=obj["object_type"],
                position=obj["position"],
                color=obj["color"],
                color_rgb=tuple(obj["color_rgb"]),
                size=obj["size"],
                pickupable=obj["pickupable"],
            )
            for obj in scene_data.get("objects", [])
        ]

        self._current_scene_config = SceneConfig(
            terrain_type=scene_data.get("terrain_type", "flat"),
            terrain_params=scene_data.get("terrain_params", {}),
            objects=obj_configs,
        )

        self._scene_objects = self._current_scene_config.get_scene_objects()

        # Build the Genesis scene
        self._build_scene(self._current_scene_config)

        # Step once to settle physics
        self._scene.step()

        # Get initial observation
        robot_state = self._read_robot_state()
        cam_rgb, cam_depth = self._capture_camera()
        return self._build_observation(robot_state, cam_rgb, cam_depth)

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Execute action in environment."""
        self._episode_steps += 1

        # Apply action (joint position targets)
        self._apply_action(action)

        # Step simulation (2 physics steps per control step)
        physics_steps = max(1, int(self.CONTROL_DT / self.SIM_DT))
        for _ in range(physics_steps):
            self._scene.step()

        # Read state
        robot_state = self._read_robot_state()
        object_states = self._read_object_states()
        cam_rgb, cam_depth = self._capture_camera()

        # Check fallen
        fallen = self._check_fallen(robot_state)

        # Check success
        if self._current_task is not None:
            self._episode_success = self._check_success(
                robot_state, object_states, self._current_task
            )

        # Compute reward
        if self._current_task is not None:
            reward = self._compute_reward(robot_state, object_states, self._current_task)
        else:
            reward = 0.0

        if fallen and not self._episode_success:
            reward = -1.0

        # Check termination
        done = self._episode_success or fallen or self._episode_steps >= self._max_episode_steps

        obs = self._build_observation(robot_state, cam_rgb, cam_depth)

        info: dict[str, Any] = {
            "task_prompt": self._current_task.task_prompt if self._current_task else "",
            "task_type": self._current_task.task_type.value if self._current_task else "",
            "episode_steps": self._episode_steps,
            "success": self._episode_success,
            "fallen": fallen,
        }

        return obs, reward, done, info

    def get_success(self) -> bool:
        """Check if task was completed successfully."""
        return self._episode_success

    def close(self) -> None:
        """Clean up environment resources."""
        if self._scene is not None:
            try:
                self._scene.destroy()
            except Exception as e:
                logger.warning("scene_destroy_failed", error=str(e))
            self._scene = None
            self._robot = None
            self._camera = None
            self._object_entities = []

    def _build_scene(self, scene_config: SceneConfig) -> None:
        """Build (or rebuild) the Genesis scene with robot, terrain, objects, camera."""
        import genesis as gs  # noqa: PLC0415

        # Destroy previous scene if exists
        if self._scene is not None:
            try:
                self._scene.destroy()
            except Exception as e:
                logger.debug("scene_destroy_on_rebuild", error=str(e))

        # Create new scene
        self._scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.SIM_DT),
            show_viewer=self._show_viewer,
            renderer=gs.renderers.Rasterizer(),
        )

        # Build terrain and objects
        if self._scene_generator is None:
            self._scene_generator = self._get_scene_generator()
        self._object_entities = self._scene_generator.build_scene(self._scene, scene_config)

        # Load robot — auto-downloads menagerie assets if needed (see menagerie.py)
        from kinitro.environments.genesis.menagerie import (  # noqa: PLC0415
            ensure_menagerie,
        )

        menagerie_path = ensure_menagerie()
        mjcf_path = os.path.join(menagerie_path, self._robot_config.mjcf_path)

        self._robot = self._scene.add_entity(
            gs.morphs.MJCF(
                file=mjcf_path,
                pos=self._robot_config.init_pos,
                quat=self._robot_config.init_quat,
            )
        )

        # Add camera (will be attached to robot head after build)
        self._camera = self._scene.add_camera(
            res=(self.IMAGE_SIZE, self.IMAGE_SIZE),
            pos=(0.0, 0.0, 1.0),  # Temporary; overwritten by attach
            lookat=(1.0, 0.0, 1.0),
            fov=90,
        )

        # Build scene (single env, no batching)
        self._scene.build(n_envs=1)

        # Attach camera to robot link for first-person view
        self._attach_ego_camera()

        # Set default pose (only actuated joints, skip 6 floating base DOFs)
        default_pos = np.array(self._robot_config.default_dof_pos, dtype=np.float32)
        n = self._robot_config.num_actuated_dofs
        actuated_dof_idx = list(range(6, 6 + n))
        self._robot.set_dofs_position(default_pos, dofs_idx_local=actuated_dof_idx)

        # Validate camera rendering works before the episode begins
        self._validate_camera()

        logger.info(
            "genesis_scene_built",
            robot=self._robot_config.name,
            terrain=scene_config.terrain_type,
            num_objects=len(scene_config.objects),
        )

    def _attach_ego_camera(self) -> None:
        """Attach camera to the robot's head/torso link for first-person POV."""
        link_name = self._robot_config.ego_camera_link
        target_link = None
        for link in self._robot.links:
            if link.name == link_name:
                target_link = link
                break

        if target_link is None:
            logger.warning("ego_camera_link_not_found", link=link_name)
            return

        # Build offset_T: 4x4 transform from camera frame to link frame.
        # Link frame: X=forward, Y=left, Z=up (MuJoCo convention).
        # Camera frame: -Z=look direction, Y=up, X=right (OpenGL convention).
        # Rotation maps camera axes → link axes:
        #   cam X (right)  → link -Y (right)
        #   cam Y (up)     → link +Z (up)
        #   cam -Z (look)  → link +X (forward)
        px, py, pz = self._robot_config.ego_camera_pos_offset
        offset = np.array(
            [
                [0.0, 0.0, -1.0, px],
                [-1.0, 0.0, 0.0, py],
                [0.0, 1.0, 0.0, pz],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        self._camera.attach(target_link, offset)
        self._camera.move_to_attach()
        logger.info(
            "ego_camera_attached", link=link_name, offset=self._robot_config.ego_camera_pos_offset
        )

    def _validate_camera(self) -> None:
        """Test-render the camera once after scene build to detect OSMesa/EGL failures early.

        Sets ``_camera_available = False`` on failure so ``_capture_camera()``
        can skip silently instead of logging 500+ identical warnings per episode.
        """
        if self._camera is None:
            self._camera_available = False
            return

        try:
            rgb, depth, _seg, _normal = self._camera.render(rgb=True, depth=True)
            if rgb is None:
                raise RuntimeError("camera.render() returned None for rgb")
            self._camera_available = True
            logger.info(
                "camera_validation_passed",
                image_size=self.IMAGE_SIZE,
                rgb_shape=rgb.shape,
            )
        except Exception as e:
            self._camera_available = False
            logger.error(
                "camera_validation_failed",
                error=repr(e),
                error_type=type(e).__name__,
                image_size=self.IMAGE_SIZE,
                hint="RGB observations will be unavailable for this episode",
                exc_info=True,
            )

    def _read_robot_state(self) -> dict[str, np.ndarray]:
        """Read robot state from Genesis tensors, convert to numpy.

        Genesis DOFs include the floating base (6 DOFs) at indices [0:6],
        followed by actuated joints at [6:6+N]. We return only the actuated
        joint positions/velocities for the observation.
        """
        pos = self._robot.get_pos().cpu().numpy().flatten()
        quat = self._robot.get_quat().cpu().numpy().flatten()
        vel = self._robot.get_vel().cpu().numpy().flatten()
        ang_vel = self._robot.get_ang().cpu().numpy().flatten()
        all_dof_pos = self._robot.get_dofs_position().cpu().numpy().flatten()
        all_dof_vel = self._robot.get_dofs_velocity().cpu().numpy().flatten()

        # Skip floating base DOFs (first 6) to get only actuated joints
        n = self._robot_config.num_actuated_dofs
        dof_pos = all_dof_pos[6 : 6 + n]
        dof_vel = all_dof_vel[6 : 6 + n]

        return {
            "base_pos": pos,  # [x, y, z]
            "base_quat": quat,  # [w, x, y, z]
            "base_vel": vel,  # [vx, vy, vz]
            "base_ang_vel": ang_vel,  # [wx, wy, wz]
            "dof_pos": dof_pos,  # [N actuated]
            "dof_vel": dof_vel,  # [N actuated]
        }

    def _read_object_states(self) -> dict[str, np.ndarray]:
        """Read positions of all tracked objects."""
        states = {}
        for i, entity in enumerate(self._object_entities):
            if i < len(self._scene_objects):
                obj_id = self._scene_objects[i].object_id
                try:
                    pos = entity.get_pos().cpu().numpy().flatten()
                    states[obj_id] = pos
                except Exception as e:
                    logger.debug("object_state_read_failed", object_id=obj_id, error=str(e))
        return states

    def _capture_camera(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Capture RGB and depth images from camera."""
        if self._camera is None or not self._camera_available:
            return None, None

        try:
            # Update camera pose to follow attached link
            if self._camera._attached_link is not None:
                self._camera.move_to_attach()

            # camera.render() returns (rgb, depth, segmentation, normal)
            rgb, depth, _seg, _normal = self._camera.render(rgb=True, depth=True)

            # Ensure uint8 for RGB
            if rgb is not None and rgb.dtype != np.uint8:
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

            return rgb, depth
        except Exception as e:
            logger.warning(
                "camera_capture_failed",
                error=repr(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            return None, None

    def _build_observation(
        self,
        robot_state: dict[str, np.ndarray],
        cam_rgb: np.ndarray | None,
        cam_depth: np.ndarray | None,
    ) -> Observation:
        """Build the full Observation from robot state and camera images."""
        rgb = {}
        if cam_rgb is not None:
            rgb["ego"] = encode_image(cam_rgb)

        depth = {}
        if cam_depth is not None:
            depth["ego"] = encode_image(cam_depth)

        return Observation(
            rgb=rgb,
            depth=depth,
            proprio={
                ProprioKeys.BASE_POS: robot_state["base_pos"].tolist(),
                ProprioKeys.BASE_QUAT: robot_state["base_quat"].tolist(),
                ProprioKeys.BASE_VEL: (
                    robot_state["base_vel"].tolist() + robot_state["base_ang_vel"].tolist()
                ),
                ProprioKeys.JOINT_POS: robot_state["dof_pos"].tolist(),
                ProprioKeys.JOINT_VEL: robot_state["dof_vel"].tolist(),
            },
            extra={
                "task_prompt": self._current_task.task_prompt if self._current_task else "",
                "task_type": self._current_task.task_type.value if self._current_task else "",
            },
        )

    def _apply_action(self, action: Action) -> None:
        """Apply joint position action to robot via PD control.

        Genesis DOFs include the floating base (6 DOFs) at indices [0:6].
        We target only the actuated joints at [6:6+N] using dofs_idx_local.
        """
        joint_action = action.get_continuous(ActionKeys.JOINT_POS_TARGET)
        if joint_action is None:
            return

        n = self._robot_config.num_actuated_dofs
        if joint_action.shape[0] < n:
            logger.warning(
                "action_undersized",
                expected=n,
                got=joint_action.shape[0],
            )
            joint_action = np.pad(joint_action, (0, n - joint_action.shape[0]))
        else:
            joint_action = joint_action[:n]

        # Clip to [-1, 1]
        joint_action = np.clip(joint_action, -1.0, 1.0)

        # Map to PD targets: target = default_pos + action * action_scale
        target_pos = self._default_dof_pos + joint_action * self._action_scale

        # Control only actuated joints (skip 6 floating base DOFs)
        actuated_dof_idx = list(range(6, 6 + n))
        self._robot.control_dofs_position(target_pos, dofs_idx_local=actuated_dof_idx)

    def _check_fallen(self, robot_state: dict[str, np.ndarray]) -> bool:
        """Check if robot has fallen over."""
        base_height = robot_state["base_pos"][2]
        return bool(base_height < self._robot_config.fall_height_threshold)
