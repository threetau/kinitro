"""Robot configuration for Genesis environments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RobotConfig:
    """Configuration for a Genesis robot.

    Captures everything that varies per robot: MJCF path, DOFs, joint names,
    default pose, action scale, capabilities. Adding a new robot = defining
    a new RobotConfig + a thin GenesisBaseEnvironment subclass.
    """

    name: str  # e.g., "g1", "go2", "franka"
    mjcf_path: str  # Path to MJCF/URDF file (relative to menagerie root)
    morph_type: str  # "mjcf" or "urdf"
    init_pos: tuple[float, float, float]  # Spawn position (x, y, z)
    init_quat: tuple[float, float, float, float]  # Spawn orientation (w, x, y, z)
    num_actuated_dofs: int  # Number of actuated joints
    joint_names: list[str]  # Ordered list of actuated joint names
    default_dof_pos: list[float]  # Default standing pose (joint angles in radians)
    action_scale: list[float]  # Per-joint action scaling
    fall_height_threshold: float  # Base height below which = fallen
    ego_camera_link: str  # Link name to attach first-person camera to
    ego_camera_pos_offset: tuple[
        float, float, float
    ]  # Camera position offset in link frame (x=fwd, y=left, z=up)

    # Task capability flags
    can_manipulate: bool = False  # Has hands/gripper
    can_locomote: bool = True  # Can walk/move
    supported_task_types: list[str] = field(default_factory=list)


# =============================================================================
# Unitree G1 Humanoid Configuration (43 actuated DOFs)
# =============================================================================
#
# Genesis loads the MJCF and reports 49 total DOFs:
#   floating_base_joint: 6 DOFs (not directly controllable)
#   + 43 actuated 1-DOF joints:
#     Legs (12): 6 per leg (hip pitch/roll/yaw, knee, ankle pitch/roll)
#     Waist (3): yaw, roll, pitch
#     Arms (14): 7 per arm (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)
#     Hands (14): 7 per hand (thumb 0/1/2, index 0/1, middle 0/1)
#
# Joint order matches Genesis MJCF parse order (verified empirically).
# Source: MuJoCo Menagerie unitree_g1/g1_with_hands.xml

G1_JOINT_NAMES = [
    # Left leg (6)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right leg (6)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist (3)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # Left arm (7)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # Right arm (7)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    # Left hand (7)
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    # Right hand (7)
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]

# Default standing pose -- slightly bent knees for stability
# fmt: off
G1_DEFAULT_DOF_POS = [
    # Left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    # Right leg
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    # Waist: yaw, roll, pitch
    0.0, 0.0, 0.0,
    # Left arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, pitch, yaw
    0.0, 0.2, 0.0, -0.3, 0.0, 0.0, 0.0,
    # Right arm
    0.0, -0.2, 0.0, -0.3, 0.0, 0.0, 0.0,
    # Left hand: thumb(3), index(2), middle(2)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    # Right hand
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]

# Per-joint action scaling (conservative for stability)
G1_ACTION_SCALE = [
    # Left leg (conservative -- large motions cause falls)
    0.3, 0.2, 0.2, 0.4, 0.3, 0.2,
    # Right leg
    0.3, 0.2, 0.2, 0.4, 0.3, 0.2,
    # Waist
    0.3, 0.2, 0.2,
    # Left arm (more freedom)
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    # Right arm
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    # Left hand
    0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
    # Right hand
    0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
]
# fmt: on

G1_CONFIG = RobotConfig(
    name="g1",
    mjcf_path="unitree_g1/g1_with_hands.xml",
    morph_type="mjcf",
    init_pos=(0.0, 0.0, 0.75),
    init_quat=(1.0, 0.0, 0.0, 0.0),
    num_actuated_dofs=43,
    joint_names=G1_JOINT_NAMES,
    default_dof_pos=G1_DEFAULT_DOF_POS,
    action_scale=G1_ACTION_SCALE,
    fall_height_threshold=0.3,
    ego_camera_link="torso_link",
    ego_camera_pos_offset=(0.15, 0.0, 0.25),  # Forward 15cm, up 25cm from torso → ~head height
    can_manipulate=True,
    can_locomote=True,
    supported_task_types=["navigate", "pickup", "place", "push"],
)
