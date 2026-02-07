"""Unitree G1 humanoid environment with navigation and manipulation tasks."""

from __future__ import annotations

import numpy as np

from kinitro.environments.genesis.base import GenesisBaseEnvironment
from kinitro.environments.genesis.robot_config import G1_CONFIG
from kinitro.environments.genesis.scene_generator import SceneGenerator
from kinitro.environments.genesis.task_generator import TaskGenerator
from kinitro.environments.genesis.task_types import TaskSpec, TaskType


class G1Environment(GenesisBaseEnvironment):
    """G1 humanoid environment with navigation and manipulation tasks.

    The Unitree G1 is a full bipedal humanoid (1.27m tall, ~35kg) with
    dexterous 3-finger hands and 37 actuated DOFs. This enables both
    locomotion and manipulation tasks.

    Supported tasks: NAVIGATE, PICKUP, PLACE, PUSH
    """

    def __init__(self, task_name: str = "g1-v0", show_viewer: bool = False) -> None:
        super().__init__(robot_config=G1_CONFIG, task_name=task_name, show_viewer=show_viewer)

    def _get_scene_generator(self) -> SceneGenerator:
        return SceneGenerator(num_objects=(3, 6))

    def _get_task_generator(self) -> TaskGenerator:
        return TaskGenerator(
            task_types=[TaskType.NAVIGATE, TaskType.PICKUP, TaskType.PLACE, TaskType.PUSH],
        )

    def _compute_reward(
        self,
        robot_state: dict[str, np.ndarray],
        object_states: dict[str, np.ndarray],
        task_spec: TaskSpec,
    ) -> float:
        """Compute reward: distance-based progress + alive bonus + task-specific bonuses."""
        reward = 0.0

        # Alive bonus (not fallen)
        reward += 0.01

        robot_pos = robot_state["base_pos"][:2]  # xy only
        target_pos = np.array(task_spec.target_position[:2], dtype=np.float32)

        if task_spec.task_type == TaskType.NAVIGATE:
            # Reward for getting closer to target
            dist = float(np.linalg.norm(robot_pos - target_pos))
            reward += max(0.0, 1.0 - dist / 5.0) * 0.1  # Progress reward

        elif task_spec.task_type == TaskType.PICKUP:
            # Reward for approaching target object
            obj_pos = object_states.get(task_spec.target_object_id)
            if obj_pos is not None:
                dist_to_obj = float(np.linalg.norm(robot_pos - obj_pos[:2]))
                reward += max(0.0, 1.0 - dist_to_obj / 5.0) * 0.05

                # Bonus if object is lifted
                initial_height = task_spec.initial_state.get("initial_height", 0.0)
                if obj_pos[2] > initial_height + 0.15:
                    reward += 1.0

        elif task_spec.task_type == TaskType.PLACE:
            obj_pos = object_states.get(task_spec.target_object_id)
            dest_pos = task_spec.destination_position
            if obj_pos is not None and dest_pos is not None:
                dist_to_dest = float(np.linalg.norm(obj_pos - np.array(dest_pos)))
                reward += max(0.0, 1.0 - dist_to_dest / 5.0) * 0.1

        elif task_spec.task_type == TaskType.PUSH:
            obj_pos = object_states.get(task_spec.target_object_id)
            dest_pos = task_spec.destination_position
            if obj_pos is not None and dest_pos is not None:
                dist_to_dest = float(np.linalg.norm(obj_pos[:2] - np.array(dest_pos[:2])))
                reward += max(0.0, 1.0 - dist_to_dest / 5.0) * 0.1

        return reward

    def _check_success(
        self,
        robot_state: dict[str, np.ndarray],
        object_states: dict[str, np.ndarray],
        task_spec: TaskSpec,
    ) -> bool:
        """Check task-specific success conditions."""
        if task_spec.task_type == TaskType.NAVIGATE:
            robot_pos = robot_state["base_pos"][:2]
            target_pos = np.array(task_spec.target_position[:2], dtype=np.float32)
            return bool(np.linalg.norm(robot_pos - target_pos) < 0.5)

        elif task_spec.task_type == TaskType.PICKUP:
            obj_pos = object_states.get(task_spec.target_object_id)
            if obj_pos is None:
                return False
            initial_height = task_spec.initial_state.get("initial_height", 0.0)
            return bool(obj_pos[2] > initial_height + 0.15)

        elif task_spec.task_type == TaskType.PLACE:
            obj_pos = object_states.get(task_spec.target_object_id)
            dest_pos = task_spec.destination_position
            if obj_pos is None or dest_pos is None:
                return False
            return bool(np.linalg.norm(obj_pos - np.array(dest_pos)) < 0.3)

        elif task_spec.task_type == TaskType.PUSH:
            obj_pos = object_states.get(task_spec.target_object_id)
            dest_pos = task_spec.destination_position
            if obj_pos is None or dest_pos is None:
                return False
            return bool(np.linalg.norm(obj_pos[:2] - np.array(dest_pos[:2])) < 0.5)

        return False
