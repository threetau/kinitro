# swarm/envs/moving_drone.py
from __future__ import annotations

from typing import cast

import gymnasium.spaces as spaces
import numpy as np
import pybullet as p
import pybullet_data
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import (
    ActionType,
    DroneModel,
    ObservationType,
    Physics,
)
from numpy.typing import NDArray

from ..constants import (
    ACTION_LAG_SEC_RANGE,
    DRAG_SCALE_RANGE,
    DRONE_HULL_RADIUS,
    DRONE_MASS,
    GOAL_TOL,
    HOVER_SEC,
    MAX_RAY_DISTANCE,
    PAYLOAD_COM_OFFSET_RANGE,
    PAYLOAD_MASS_FACTOR_RANGE,
    THRUST_SCALE_RANGE,
    WIND_XY_RANGE,
)

# -- project-level utilities ------------------------------------------------
from ..validator.reward import flight_reward  # 3-term scorer
from .env_builder import build_world


class MovingDroneAviary(BaseRLAviary):
    """
    Single-drone environment whose *start*, *goal* and *horizon* are supplied
    via an external `MapTask`.

    The per-step reward is the **increment** of `flight_reward`, so it can be
    fed directly to PPO/TD3/etc. without extra shaping.
    """

    MAX_TILT_RAD: float = 1.047  # safety cut-off for roll / pitch (rad)

    # --------------------------------------------------------------------- #
    # 1. constructor
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        task,
        drone_model: DroneModel = DroneModel.CF2X,
        physics: Physics = Physics.PYB,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        gui: bool = False,
        record: bool = False,
        obs: ObservationType = ObservationType.KIN,
        act: ActionType = ActionType.RPM,
    ):
        """
        Parameters
        ----------
        task : MapTask
            Must expose `.start`, `.goal`, `.horizon`, `.sim_dt`.
        Remaining arguments are forwarded to ``BaseRLAviary`` unchanged.
        """
        self.task = task
        self.GOAL_POS = np.asarray(task.goal, dtype=float)
        self.EP_LEN_SEC = float(task.horizon)

        # internal book-keeping
        self._time_alive = 0.0
        self._hover_sec = 0.0
        self._success = False
        self._collision = False
        self._t_to_goal = None
        self._prev_score = 0.0
        self.payload_enabled = bool(getattr(task, "payload_enabled", False))
        self.domain_randomization = bool(
            getattr(task, "domain_randomization", self.payload_enabled)
        )

        # Payload/randomization parameters (set from task)
        if self.payload_enabled:
            self.payload_mass_factor = float(
                getattr(task, "payload_mass_factor", 1.0) or 1.0
            )
            self.payload_com_offset = np.asarray(
                getattr(task, "payload_com_offset", (0.0, 0.0, 0.0)), dtype=float
            )
            self.thrust_scale = float(getattr(task, "thrust_scale", 1.0) or 1.0)
            self.drag_scale = float(getattr(task, "drag_scale", 1.0) or 1.0)
            wind_xy = getattr(task, "wind_xy", (0.0, 0.0)) or (0.0, 0.0)
            # Treat wind_xy as a desired acceleration (m/s^2) and convert to force.
            base_mass = float(getattr(self, "M", DRONE_MASS))
            self.wind_force = (
                np.array([wind_xy[0], wind_xy[1], 0.0], dtype=float) * base_mass
            )
            self.action_latency = float(getattr(task, "action_latency", 0.0) or 0.0)
        else:
            self.payload_mass_factor = 1.0
            self.payload_com_offset = np.array([0.0, 0.0, 0.0], dtype=float)
            self.thrust_scale = 1.0
            self.drag_scale = 1.0
            self.wind_force = np.array([0.0, 0.0, 0.0], dtype=float)
            self.action_latency = 0.0

        # Track payload bodies for cleanup
        self._payload_body_id: int | None = None
        self._payload_constraint_id: int | None = None
        self._prev_action = None

        # --- define 16 ray directions for obstacle detection ---
        self._init_ray_directions()

        # Let BaseRLAviary set up the PyBullet world
        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=np.asarray([task.start]),
            initial_rpys=None,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=obs,
            act=act,
        )

        # --- extend observation with obstacle distances (16-D) + goal vector (3-D) ---
        obs_space = cast(spaces.Box, self.observation_space)
        old_low, old_high = obs_space.low, obs_space.high

        # Distance sensors: 16 dimensions, range [0.0, 1.0] (scaled from meters)
        dist_low = np.zeros((old_low.shape[0], 16), dtype=np.float32)
        dist_high = np.ones(
            (old_high.shape[0], 16), dtype=np.float32
        )  # scaled to [0.0, 1.0]

        # Goal vector: 3 dimensions, unlimited range
        goal_low = -np.ones((old_low.shape[0], 3), dtype=np.float32) * np.inf
        goal_high = +np.ones((old_high.shape[0], 3), dtype=np.float32) * np.inf

        if self.payload_enabled or self.domain_randomization:
            extra_low = np.array(
                [
                    PAYLOAD_MASS_FACTOR_RANGE[0],
                    -PAYLOAD_COM_OFFSET_RANGE[0],
                    -PAYLOAD_COM_OFFSET_RANGE[1],
                    -PAYLOAD_COM_OFFSET_RANGE[2],
                    THRUST_SCALE_RANGE[0],
                    DRAG_SCALE_RANGE[0],
                    WIND_XY_RANGE[0],
                    WIND_XY_RANGE[0],
                    ACTION_LAG_SEC_RANGE[0],
                ],
                dtype=np.float32,
            ).reshape(1, -1)
            extra_high = np.array(
                [
                    PAYLOAD_MASS_FACTOR_RANGE[1],
                    PAYLOAD_COM_OFFSET_RANGE[0],
                    PAYLOAD_COM_OFFSET_RANGE[1],
                    PAYLOAD_COM_OFFSET_RANGE[2],
                    THRUST_SCALE_RANGE[1],
                    DRAG_SCALE_RANGE[1],
                    WIND_XY_RANGE[1],
                    WIND_XY_RANGE[1],
                    ACTION_LAG_SEC_RANGE[1],
                ],
                dtype=np.float32,
            ).reshape(1, -1)

            low = np.concatenate([old_low, dist_low, goal_low, extra_low], axis=1)
            high = np.concatenate([old_high, dist_high, goal_high, extra_high], axis=1)
        else:
            low = np.concatenate([old_low, dist_low, goal_low], axis=1)
            high = np.concatenate([old_high, dist_high, goal_high], axis=1)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # --------------------------------------------------------------------- #
    # 2. low-level helpers
    # --------------------------------------------------------------------- #
    @property
    def _sim_dt(self) -> float:
        """Physics step in seconds (1 / CTRL_FREQ)."""
        return 1.0 / self.CTRL_FREQ

    def _init_ray_directions(self):
        """Initialize 16 ray directions for obstacle detection - balanced with side coverage."""
        # Mathematical constants
        cos_45 = np.cos(np.radians(45))  # sqrt2/2 ~ 0.707107
        sin_45 = np.sin(np.radians(45))  # sqrt2/2 ~ 0.707107
        cos_30 = np.cos(np.radians(30))  # sqrt3/2 ~ 0.866025
        sin_30 = np.sin(np.radians(30))  # 1/2 = 0.500000

        self.ray_directions = np.array(
            [
                # --- 2 PURE VERTICAL (essential for landing/overhead) ---
                [0, 0, 1],  # 1: Pure Up
                [0, 0, -1],  # 2: Pure Down
                # --- 8 HORIZONTAL - BALANCED COVERAGE ---
                [1, 0, 0],  # 3: Forward (0 degrees)
                [cos_45, sin_45, 0],  # 4: Forward-Right (45 degrees)
                [0, 1, 0],  # 5: Right (90 degrees)
                [-cos_45, sin_45, 0],  # 6: Back-Right (135 degrees)
                [-1, 0, 0],  # 7: Back (180 degrees)
                [-cos_45, -sin_45, 0],  # 8: Back-Left (225 degrees)
                [0, -1, 0],  # 9: Left (270 degrees)
                [cos_45, -sin_45, 0],  # 10: Forward-Left (315 degrees)
                # --- 6 DIAGONAL RAYS (3D coverage at +/-30 degrees elevation) ---
                [cos_30, 0, sin_30],  # 11: Forward-Up (30 degrees)
                [cos_30, 0, -sin_30],  # 12: Forward-Down (-30 degrees)
                [-cos_30, 0, sin_30],  # 13: Back-Up (30 degrees)
                [-cos_30, 0, -sin_30],  # 14: Back-Down (-30 degrees)
                [0, cos_30, sin_30],  # 15: Right-Up (30 degrees)
                [0, -cos_30, sin_30],  # 16: Left-Up (30 degrees)
            ],
            dtype=np.float32,
        )

        # Detection range and small origin offset to avoid self-hits
        self.max_ray_distance: float = MAX_RAY_DISTANCE
        self._ray_origin_offset: float = DRONE_HULL_RADIUS

    def _get_obstacle_distances(
        self, drone_position: np.ndarray, drone_orientation: np.ndarray
    ) -> np.ndarray:
        """
        Perform 16-ray casting for obstacle detection using batch processing.

        Parameters
        ----------
        drone_position : np.ndarray
            Current drone position [x, y, z] in world frame
        drone_orientation : np.ndarray
            Current drone orientation as 3x3 rotation matrix (body->world)

        Returns
        -------
        np.ndarray
            Array of 16 distances in meters [0.0 - MAX_RAY_DISTANCE].
            Distances are measured from the drone COM and clamped to max range.
        """
        rot_matrix = drone_orientation

        start_positions = []
        end_positions = []

        # Length of the ray segment when starting offset outside the hull
        seg_len = self.max_ray_distance - self._ray_origin_offset
        if seg_len <= 0:
            seg_len = 1e-6

        for direction in self.ray_directions:
            # Transform direction from body frame to world frame and normalize
            world_dir = rot_matrix @ direction
            n = float(np.linalg.norm(world_dir))
            if n < 1e-9:
                world_dir = np.array([0.0, 0.0, 1.0], dtype=float)  # fallback
            else:
                world_dir = world_dir / n

            # Start just outside the drone to avoid self-hit, end at max range from COM
            start_pos = drone_position + world_dir * self._ray_origin_offset
            end_pos = drone_position + world_dir * self.max_ray_distance

            start_positions.append(start_pos.tolist())
            end_positions.append(end_pos.tolist())

        # Batch ray test - pass the correct physics client
        results = p.rayTestBatch(
            start_positions, end_positions, physicsClientId=getattr(self, "CLIENT", 0)
        )

        # Extract distances from results, converting "from start" to "from COM"
        distances = []
        for r in results:
            hit_uid = r[0]  # -1 means no hit
            hit_frac = float(r[2])  # [0,1] along the segment (start->end)
            if hit_uid != -1:
                # distance from COM = offset + fraction*segment_length, clamped
                d = self._ray_origin_offset + hit_frac * seg_len
                distances.append(float(min(self.max_ray_distance, max(0.0, d))))
            else:
                distances.append(self.max_ray_distance)

        return np.array(distances, dtype=np.float32)

    # ---------------------- payload / wind helpers ---------------------- #
    def _clear_payload(self) -> None:
        """Remove attached payload if present."""
        cid = self._payload_constraint_id
        if cid is not None:
            try:
                # Skip removal if constraint is already gone
                p.getConstraintInfo(cid)
                p.removeConstraint(cid)
            except Exception as exc:
                print(f"[WARN] payload clear: failed to remove constraint {cid}: {exc}")
        self._payload_constraint_id = None

        bid = self._payload_body_id
        if bid is not None:
            try:
                p.getBodyInfo(bid)
                p.removeBody(bid)
            except Exception as exc:
                print(f"[WARN] payload clear: failed to remove body {bid}: {exc}")
        self._payload_body_id = None

    def _attach_payload(self) -> None:
        """Attach a simple payload mass via a fixed constraint to shift COM."""
        self._clear_payload()
        if self.payload_mass_factor <= 1.0:
            return

        base_mass = float(getattr(self, "M", DRONE_MASS))
        payload_mass = max(0.0, self.payload_mass_factor - 1.0) * base_mass
        if payload_mass <= 0:
            return

        # Create a small sphere payload and place it below the drone to avoid collisions
        col = p.createCollisionShape(
            p.GEOM_SPHERE, radius=0.05, physicsClientId=self.CLIENT
        )
        drone_pos, _ = p.getBasePositionAndOrientation(
            self.DRONE_IDS[0], physicsClientId=self.CLIENT
        )
        payload_offset = np.asarray(self.payload_com_offset, dtype=float)
        if abs(payload_offset[2]) < 0.05:
            payload_offset[2] = -0.1
        payload_pos = np.asarray(drone_pos, dtype=float) + payload_offset
        payload_body = p.createMultiBody(
            baseMass=payload_mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=-1,
            basePosition=payload_pos.tolist(),
            physicsClientId=self.CLIENT,
        )

        self._payload_body_id = payload_body
        # Fixed constraint to drone with offset
        self._payload_constraint_id = p.createConstraint(
            parentBodyUniqueId=self.DRONE_IDS[0],
            parentLinkIndex=-1,
            childBodyUniqueId=payload_body,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=payload_offset.tolist(),
            childFramePosition=[0, 0, 0],
            physicsClientId=self.CLIENT,
        )
        # Disable collisions between payload and drone to avoid instant contact failures
        try:
            p.setCollisionFilterPair(
                bodyUniqueIdA=self.DRONE_IDS[0],
                bodyUniqueIdB=payload_body,
                linkIndexA=-1,
                linkIndexB=-1,
                enableCollision=0,
                physicsClientId=self.CLIENT,
            )
        except Exception:
            pass

    def _apply_wind_force(self) -> None:
        """Apply a constant lateral wind force to the drone base."""
        if np.allclose(self.wind_force, 0.0):
            return
        try:
            p.applyExternalForce(
                objectUniqueId=self.DRONE_IDS[0],
                linkIndex=-1,
                forceObj=self.wind_force.tolist(),
                posObj=[0, 0, 0],
                flags=p.WORLD_FRAME,
                physicsClientId=self.CLIENT,
            )
        except Exception:
            pass

    def _euler_to_rotation_matrix(
        self, roll: float, pitch: float, yaw: float
    ) -> np.ndarray:
        """
        Convert Euler angles directly to rotation matrix (body->world).
        """
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ]
        )

    def _check_collision(self) -> bool:
        """
        Check if drone has collided with any obstacle using PyBullet contact points.
        Returns True if collision detected, False otherwise.

        Allowed contacts: landing surface and platform support cylinder.
        Collision contacts: obstacles, walls, ground plane.
        """
        drone_id = self.DRONE_IDS[0]
        contact_points = p.getContactPoints(
            bodyA=drone_id, physicsClientId=getattr(self, "CLIENT", 0)
        )

        if not contact_points:
            return False

        landing_surface_uid = getattr(self, "_landing_surface_uid", None)
        platform_support_uid = getattr(self, "_platform_support_uid", None)

        for contact in contact_points:
            body_b = contact[2]
            if body_b != -1:
                normal_force = contact[9]
                if normal_force > 0.01:
                    if (
                        landing_surface_uid is not None
                        and body_b == landing_surface_uid
                    ):
                        continue
                    if (
                        platform_support_uid is not None
                        and body_b == platform_support_uid
                    ):
                        continue
                    return True

        return False

    # --------------------------------------------------------------------- #
    # 3. OpenAI-Gym API overrides
    # --------------------------------------------------------------------- #
    def reset(self, **kwargs):
        """
        Resets the underlying simulator and internal counters,
        returns initial observation and info as usual.
        """
        obs, info = super().reset(**kwargs)

        # Rebuild world after every reset to ensure obstacles/platform are present
        cli = getattr(self, "CLIENT", 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        platform_support_uid, landing_surface_uid = build_world(
            seed=self.task.map_seed,
            cli=cli,
            start=self.task.start,
            goal=self.task.goal,
            challenge_type=self.task.challenge_type,
        )
        self._platform_support_uid = platform_support_uid
        self._landing_surface_uid = landing_surface_uid

        # Respawn drone at requested start pose
        start_xyz = np.asarray(self.task.start, dtype=float)
        start_quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        p.resetBasePositionAndOrientation(
            self.DRONE_IDS[0],
            start_xyz,
            start_quat,
            physicsClientId=cli,
        )
        # Reset velocity
        p.resetBaseVelocity(
            self.DRONE_IDS[0], [0, 0, 0], [0, 0, 0], physicsClientId=cli
        )

        # Apply drag scaling if enabled
        if self.payload_enabled and self.drag_scale != 1.0:
            try:
                p.changeDynamics(
                    self.DRONE_IDS[0],
                    -1,
                    linearDamping=0.01 * self.drag_scale,
                    angularDamping=0.01 * self.drag_scale,
                    physicsClientId=cli,
                )
            except Exception:
                pass

        # Attach payload if enabled
        if self.payload_enabled:
            self._attach_payload()
        self._prev_action = None

        self._time_alive = 0.0
        self._hover_sec = 0.0
        self._success = False
        self._collision = False
        self._t_to_goal = None

        # baseline score (t = 0, e = 0)
        self._prev_score = flight_reward(
            success=False,
            t=0.0,
            horizon=self.EP_LEN_SEC,
            task=None,
        )

        return obs, info

    # -------- reward ----------------------------------------------------- #
    def _computeReward(self) -> float:  # noqa: N802
        """
        **Incremental** reward based on the three-term `flight_reward`.
        """
        # Apply wind each step before computing reward (for next physics step)
        # TODO: disabled for now - needs more work.
        # if self.payload_enabled:
        #     self._apply_wind_force()

        # current distance to goal
        state = self._getDroneStateVector(0)
        float(np.linalg.norm(state[0:3] - self.GOAL_POS))

        # -- success detection: remain inside TAO badge with proper height constraints --
        # Use 2D horizontal distance + vertical constraints
        horizontal_distance = float(
            np.linalg.norm(state[0:2] - self.GOAL_POS[0:2])
        )  # X,Y only
        vertical_distance = abs(state[2] - self.GOAL_POS[2])  # Z only

        # Success requires: within TAO badge horizontally + proper height + above platform
        reached = (
            horizontal_distance < GOAL_TOL  # Within TAO badge radius
            and vertical_distance < 0.3  # Within 30cm of surface
            and state[2] >= self.GOAL_POS[2] - 0.1
        )  # Above platform (not below)
        if reached:
            self._hover_sec += self._sim_dt
            if self._hover_sec >= HOVER_SEC and not self._success:
                self._success = True
                self._t_to_goal = self._time_alive
        else:
            self._hover_sec = 0.0

        # -- clock update ----------------------------------------------------
        self._time_alive += self._sim_dt

        # -- call new reward function ---------------------------------------
        # If collision detected, force score to 0
        if self._collision:
            score = 0.0
        else:
            score = flight_reward(
                success=self._success,
                t=(self._t_to_goal if self._success else self._time_alive),
                horizon=self.EP_LEN_SEC,
                task=None,
            )

        r_t = score - self._prev_score
        self._prev_score = score
        return float(r_t)

    # -------- action preprocessing --------------------------------------- #
    def _preprocessAction(self, action):  # noqa: N802
        """Inject simple action lag and thrust scaling before base handling."""
        processed = super()._preprocessAction(action)

        if not (self.payload_enabled or self.domain_randomization):
            return processed

        if self._prev_action is None:
            self._prev_action = np.array(processed, copy=True)

        if self.action_latency > 0:
            smoothing = np.clip(self.action_latency / max(self._sim_dt, 1e-3), 0.0, 0.9)
            processed = (1 - smoothing) * processed + smoothing * self._prev_action

        # Thrust scaling for velocity commands (approximate)
        processed = processed * self.thrust_scale
        self._prev_action = np.array(processed, copy=True)
        return processed

    # -------- termination ------------------------------------------------ #
    def _computeTerminated(self) -> bool:  # noqa: N802
        """
        Episode ends when success condition is met OR collision detected.
        """
        # Check for collision first
        if self._check_collision():
            self._collision = True
            return True

        # Check for success
        return bool(self._success)

    # -------- truncation (timeout / safety) ------------------------------ #
    def _computeTruncated(self) -> bool:  # noqa: N802
        """
        Early termination on excessive tilt or elapsed horizon.
        """
        # safety cut-off
        state = self._getDroneStateVector(0)
        roll, pitch = state[7], state[8]
        if abs(roll) > self.MAX_TILT_RAD or abs(pitch) > self.MAX_TILT_RAD:
            return True

        # timeout
        return self._time_alive >= self.EP_LEN_SEC

    # -------- extra logging --------------------------------------------- #
    def _computeInfo(self):  # noqa: N802
        state = self._getDroneStateVector(0)
        dist = float(np.linalg.norm(state[0:3] - self.GOAL_POS))
        return {
            "distance_to_goal": dist,
            "score": self._prev_score,
            "success": self._success,
            "collision": self._collision,
            "t_to_goal": self._t_to_goal,
        }

    # -------- observation extension -------------------------------------- #
    def _computeObs(self) -> np.ndarray:  # noqa: N802
        """
        Full base observation (112-D) + obstacle distances (16-D) + goal vector (3-D) -> 131-D.

        Distances are computed from the **current PyBullet pose** and
        then scaled to [0,1] by dividing by `max_ray_distance` (10 m).
        """
        base_obs: NDArray[np.float32] | None = (
            super()._computeObs()
        )  # shape (1, 112) in your setup
        if base_obs is None:
            return np.zeros((1, 131), dtype=np.float32)

        # --- Get exact pose from PyBullet to stay in sync with physics ---
        uid = self.DRONE_IDS[0]
        pos_w, orn_w = p.getBasePositionAndOrientation(
            uid, physicsClientId=getattr(self, "CLIENT", 0)
        )
        pos_w = np.asarray(pos_w, dtype=float)
        rot_m = np.asarray(p.getMatrixFromQuaternion(orn_w), dtype=float).reshape(3, 3)

        # --- Cast rays from that pose ---
        distances_m = self._get_obstacle_distances(pos_w, rot_m).reshape(1, 16)

        # Scale to [0,1] for the observation
        distances_scaled = distances_m / self.max_ray_distance

        # Goal vector relative to current position (scaled by ray distance)
        rel = ((self.GOAL_POS - pos_w) / self.max_ray_distance).reshape(1, 3)

        parts = [base_obs, distances_scaled, rel]
        if self.payload_enabled or self.domain_randomization:
            extras = np.array(
                [
                    self.payload_mass_factor,
                    self.payload_com_offset[0],
                    self.payload_com_offset[1],
                    self.payload_com_offset[2],
                    self.thrust_scale,
                    self.drag_scale,
                    self.wind_force[0],
                    self.wind_force[1],
                    self.action_latency,
                ],
                dtype=np.float32,
            ).reshape(1, -1)
            parts.append(extras)

        return np.concatenate(parts, axis=1).astype(np.float32)

    # -------- vision helper ----------------------------------------------- #
    def get_third_person_rgb(
        self,
        distance: float = 5.0,
        yaw_deg: float = -45.0,
        pitch_deg: float = -30.0,
        fov: float = 60.0,
        aspect: float = 1.0,
    ) -> np.ndarray:
        """
        Capture an RGB(A) image from a fixed third-person camera aimed at the drone.

        The camera is placed `distance` meters away from the drone at the
        specified yaw/pitch and looks at the drone center with an up-vector of +Z.
        """
        client = self.CLIENT
        pos_w, _ = p.getBasePositionAndOrientation(
            self.DRONE_IDS[0], physicsClientId=client
        )
        pos_w = np.asarray(pos_w, dtype=float)

        # Compute camera eye position from spherical offsets
        yaw = np.deg2rad(yaw_deg)
        pitch = np.deg2rad(pitch_deg)
        cam_dir = np.array(
            [
                np.cos(pitch) * np.cos(yaw),
                np.cos(pitch) * np.sin(yaw),
                np.sin(pitch),
            ],
            dtype=float,
        )
        eye = pos_w - cam_dir * distance

        view = p.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=pos_w.tolist(),
            cameraUpVector=[0, 0, 1],
            physicsClientId=client,
        )
        res = getattr(self, "IMG_RES", None)
        if res is None:
            res = np.array([320, 240], dtype=int)
        width, height = int(res[0]), int(res[1])
        proj = p.computeProjectionMatrixFOV(
            fov=fov, aspect=aspect, nearVal=self.L, farVal=1000.0
        )
        _, _, rgb, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view,
            projectionMatrix=proj,
            flags=p.ER_NO_SEGMENTATION_MASK,
            physicsClientId=client,
            renderer=p.ER_TINY_RENDERER,
        )
        rgba = np.reshape(rgb, (height, width, 4)).astype(np.uint8)
        return rgba
