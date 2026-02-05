# Kinitro Evaluation Environments

This directory contains the affinetes-compatible evaluation environments for the Kinitro robotics subnet. Each environment family has its own directory with a Dockerfile, Actor class (`env.py`), and dependencies (`requirements.txt`).

## Directory Structure

Each environment family directory must contain:

```
<family_name>/
├── Dockerfile          # Container build instructions
├── env.py              # Actor class with evaluate() method
├── requirements.txt    # Python dependencies
└── metadata.json       # Family display name and description
```

## Building Environment Images

Use the `kinitro env build` command to build environment-specific Docker images:

```bash
# Build MetaWorld environment (~1GB image)
kinitro env build metaworld --tag kinitro/metaworld:v1

# Build ProcTHOR environment (~3GB image)
kinitro env build procthor --tag kinitro/procthor:v1

# Build and push to registry
kinitro env build metaworld --push --registry docker.io/myuser
```

## Environment Families

### MetaWorld

MuJoCo-based manipulation tasks for robot arm control.

**Supported environments:**

- `metaworld/reach-v3` - Move end-effector to target position
- `metaworld/push-v3` - Push object to goal location
- `metaworld/pick-place-v3` - Pick up object and place at target
- `metaworld/door-open-v3` - Open a door
- `metaworld/drawer-open-v3` - Open a drawer
- `metaworld/drawer-close-v3` - Close a drawer
- `metaworld/button-press-v3` - Press a button from top-down
- `metaworld/peg-insert-v3` - Insert peg into hole

**Platform:** Works on any platform (Linux, macOS, Windows)

**Image size:** ~1GB

### ProcTHOR

AI2-THOR procedural house environments for embodied AI tasks.

**Supported environments:**

- `procthor/v0` - Procedural house tasks with task types:
  - PICKUP - Pick up an object
  - PLACE - Place an object at a location
  - OPEN - Open a container/door
  - CLOSE - Close a container/door
  - TOGGLE_ON - Turn on an appliance
  - TOGGLE_OFF - Turn off an appliance

**Platform:** Requires native x86_64 Linux (does NOT work on ARM64 or under emulation)

**Image size:** ~3GB (includes pre-downloaded AI2-THOR binaries)

## Backend Configuration

Configure your backend to use the appropriate image for each environment family. Set these environment variables:

```bash
KINITRO_BACKEND_EVAL_IMAGE_METAWORLD=kinitro/metaworld:v1
KINITRO_BACKEND_EVAL_IMAGE_PROCTHOR=kinitro/procthor:v1
```

---

# Guide: Adding New Environments

This guide explains how to add support for new robotics environments to Kinitro. Follow these steps to integrate a new simulation platform (e.g., a robot arm manipulation environment, a mobile robot navigation environment, etc.).

## Overview

Adding a new environment involves:

1. **Creating the environment directory** with required files
2. **Implementing the `RoboticsEnvironment` class** with task generation, reset, step, and success logic
3. **Defining task types and specifications** for procedural generation
4. **Implementing reward signals** (sparse or dense)
5. **Creating the container Actor class** for affinetes evaluation
6. **Registering the environment** in the registry

## Step 1: Create the Environment Directory

Create a new directory under `environments/`:

```
environments/
└── myrobot/
    ├── Dockerfile          # Container build spec
    ├── env.py              # Actor class for affinetes
    ├── requirements.txt    # Python dependencies
    └── metadata.json       # Display info
```

### metadata.json

```json
{
  "name": "MyRobot",
  "description": "Robot arm manipulation with procedural task generation"
}
```

## Step 2: Implement the RoboticsEnvironment Class

Create your environment class in `kinitro/environments/myrobot/environment.py` inheriting from `RoboticsEnvironment`:

```python
from kinitro.environments.base import RoboticsEnvironment, TaskConfig
from kinitro.rl_interface import Observation, Action, ProprioKeys, ActionKeys, coerce_action

class MyRobotEnvironment(RoboticsEnvironment):
    """Your environment implementation."""

    def __init__(self, task_name: str = "myrobot-v0"):
        self._task_name = task_name
        # Initialize your simulator here

    @property
    def env_name(self) -> str:
        """Environment family name."""
        return "myrobot"

    @property
    def task_name(self) -> str:
        """Specific task variant."""
        return self._task_name
```

### Required Abstract Methods

You must implement these methods from `RoboticsEnvironment`:

| Method | Purpose |
|--------|---------|
| `generate_task(seed: int) -> TaskConfig` | Create procedural task from seed |
| `reset(task_config: TaskConfig) -> Observation` | Initialize environment, return first observation |
| `step(action: Action) -> tuple[obs, reward, done, info]` | Execute action, return results |
| `get_success() -> bool` | Check if task completed successfully |

## Step 3: Define Task Configuration

Tasks are specified using `TaskConfig`, which contains procedural parameters:

```python
from kinitro.environments.base import TaskConfig
import numpy as np

def generate_task(self, seed: int) -> TaskConfig:
    """Generate a procedural task from seed."""
    rng = np.random.default_rng(seed)

    # Randomize object positions
    object_positions = np.array([
        rng.uniform(-0.3, 0.3),  # x
        rng.uniform(0.0, 0.2),   # y
        rng.uniform(-0.3, 0.3),  # z
    ])

    # Randomize target positions
    target_positions = np.array([
        rng.uniform(-0.3, 0.3),
        rng.uniform(0.0, 0.2),
        rng.uniform(-0.3, 0.3),
    ])

    # Physics randomization (optional)
    physics_params = {
        "friction": rng.uniform(0.3, 0.7),
        "damping": rng.uniform(0.1, 0.3),
        "mass_scale": rng.uniform(0.8, 1.2),
    }

    # Domain randomization (visual, etc.)
    domain_randomization = {
        "camera_fov": rng.uniform(60, 90),
        "lighting_intensity": rng.uniform(0.5, 1.5),
        "texture_variation": int(rng.integers(0, 10)),
    }

    return TaskConfig(
        env_name=self.env_name,
        task_name=self._task_name,
        seed=seed,
        object_positions=object_positions,
        target_positions=target_positions,
        physics_params=physics_params,
        domain_randomization=domain_randomization,
    )
```

### TaskConfig Fields

| Field | Type | Purpose |
|-------|------|---------|
| `env_name` | str | Family name (e.g., "myrobot") |
| `task_name` | str | Task variant (e.g., "pick-place-v1") |
| `seed` | int | Random seed for reproducibility |
| `object_positions` | np.ndarray | Randomized object placements |
| `target_positions` | np.ndarray | Randomized goal positions |
| `physics_params` | dict | Physics randomization (friction, mass, etc.) |
| `domain_randomization` | dict | Visual/domain randomization |

## Step 4: Define Task Types (for Complex Environments)

For environments with multiple task types (like ProcTHOR), define a task type enum and task spec:

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Any

class TaskType(Enum):
    """Types of tasks the agent can perform."""
    PICK = "pick"           # Pick up an object
    PLACE = "place"         # Place object at location
    SLICE = "slice"         # Cut an object
    STACK = "stack"         # Stack objects
    POUR = "pour"           # Pour contents

@dataclass
class TaskSpec:
    """Specification for a procedurally generated task."""
    task_type: TaskType
    task_prompt: str                    # Natural language instruction
    target_object_id: str               # Object to manipulate
    target_object_type: str             # Object class (e.g., "Apple")
    destination_object_id: str | None = None   # For PLACE tasks
    destination_object_type: str | None = None
    initial_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type.value,
            "task_prompt": self.task_prompt,
            "target_object_id": self.target_object_id,
            "target_object_type": self.target_object_type,
            "destination_object_id": self.destination_object_id,
            "destination_object_type": self.destination_object_type,
            "initial_state": self.initial_state,
        }
```

### Natural Language Task Prompts

Generate varied natural language instructions:

```python
PROMPT_TEMPLATES = {
    TaskType.PICK: [
        "Pick up the {object}.",
        "Grab the {object}.",
        "Take the {object}.",
    ],
    TaskType.SLICE: [
        "Slice the {object} into pieces.",
        "Cut the {object}.",
        "Chop up the {object}.",
    ],
    TaskType.PLACE: [
        "Put the {object} on the {destination}.",
        "Place the {object} on the {destination}.",
        "Move the {object} to the {destination}.",
    ],
}

def generate_prompt(task_type: TaskType, target: str,
                    destination: str | None, rng) -> str:
    templates = PROMPT_TEMPLATES[task_type]
    template = templates[rng.integers(0, len(templates))]
    if destination:
        return template.format(object=target, destination=destination)
    return template.format(object=target)
```

## Step 5: Specify Reward Signals

Kinitro supports two reward structures:

### Sparse Rewards (Recommended for Evaluation)

Sparse rewards give signal only at task completion. This is the default for Kinitro evaluation:

```python
def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
    # Execute action in simulator
    self._simulator.step(action)
    self._episode_steps += 1

    # Update success status
    self._check_success()

    # Sparse reward: 1.0 on success, 0.0 otherwise
    if self._episode_success:
        reward = 1.0
    else:
        reward = 0.0

    done = self._episode_success or self._episode_steps >= self._max_steps

    return self._build_observation(), reward, done, {"success": self._episode_success}
```

**Evaluation score**: Final score is `1.0` if `get_success()` returns `True`, else `0.0`.

### Dense Rewards (for Training Reference)

Dense rewards provide incremental feedback for progress. Useful if you want to provide training signals:

```python
def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
    # Execute action
    prev_distance = self._distance_to_goal()
    self._simulator.step(action)
    curr_distance = self._distance_to_goal()

    # Dense reward components
    reward = 0.0

    # Progress reward: getting closer to goal
    progress = prev_distance - curr_distance
    reward += progress * 10.0  # Scale factor

    # Grasp reward: successfully grasping target
    if self._just_grasped_target():
        reward += 0.5

    # Completion bonus
    self._check_success()
    if self._episode_success:
        reward += 1.0

    # Safety penalty (optional)
    if self._collision_detected():
        reward -= 0.1

    done = self._episode_success or self._episode_steps >= self._max_steps

    return self._build_observation(), reward, done, {
        "success": self._episode_success,
        "distance_to_goal": curr_distance,
    }
```

**Important**: Even with dense rewards, the **evaluation score** is still binary based on `get_success()`. Dense rewards are for policy training, not evaluation scoring.

## Step 6: Define Success Criteria

Implement `get_success()` to return whether the task objective is achieved:

```python
def get_success(self) -> bool:
    """Check if task completed successfully."""
    return self._episode_success

def _check_success(self) -> None:
    """Update success status based on task type."""
    if self._current_task is None:
        self._episode_success = False
        return

    task = self._current_task

    if task.task_type == TaskType.PICK:
        # Success: target object is being held
        self._episode_success = self._holding_object == task.target_object_id

    elif task.task_type == TaskType.PLACE:
        # Success: object is on destination receptacle
        obj = self._get_object(task.target_object_id)
        if obj and task.destination_object_id:
            self._episode_success = (
                task.destination_object_id in obj.parent_receptacles
            )

    elif task.task_type == TaskType.SLICE:
        # Success: object has been sliced (state changed)
        obj = self._get_object(task.target_object_id)
        self._episode_success = obj is not None and obj.is_sliced

    elif task.task_type == TaskType.STACK:
        # Success: objects stacked in correct order
        self._episode_success = self._check_stack_order()
```

### Success Criteria Guidelines

| Task Type | Success Condition |
|-----------|-------------------|
| PICK | Target object in gripper |
| PLACE | Object on destination receptacle |
| OPEN | Container/door is open |
| CLOSE | Container/door is closed |
| TOGGLE_ON | Device is powered on |
| TOGGLE_OFF | Device is powered off |
| SLICE | Object has been cut |
| STACK | Objects in correct vertical order |
| POUR | Container emptied into target |

## Step 7: Implement the Observation Interface

Return observations using `Observation` with dict-based proprioceptive channels:

```python
from kinitro.rl_interface import Observation, ProprioKeys

def _build_observation(self) -> Observation:
    """Build observation from simulator state."""

    # Get end-effector state from simulator
    ee_pos = self._simulator.get_ee_position()      # [x, y, z]
    ee_quat = self._simulator.get_ee_orientation()  # [x, y, z, w]
    ee_lin_vel = self._simulator.get_ee_linear_velocity()
    ee_ang_vel = self._simulator.get_ee_angular_velocity()
    gripper_state = self._simulator.get_gripper_state()  # 0.0-1.0

    # Get camera images
    rgb_images = {}
    rgb_images["ego"] = self._simulator.render_camera("ego_cam")  # np.ndarray
    if self._use_wrist_camera:
        rgb_images["wrist"] = self._simulator.render_camera("wrist_cam")

    # Optional depth
    depth = None
    if self._use_depth:
        depth = {"ego": self._simulator.render_depth("ego_cam")}

    return Observation(
        proprio={
            ProprioKeys.EE_POS: ee_pos.tolist(),
            ProprioKeys.EE_QUAT: ee_quat.tolist(),
            ProprioKeys.EE_VEL_LIN: ee_lin_vel.tolist(),
            ProprioKeys.EE_VEL_ANG: ee_ang_vel.tolist(),
            ProprioKeys.GRIPPER: [float(gripper_state)],
        },
        rgb=rgb_images,  # Will be base64 encoded automatically
        depth=depth,
        cam_intrinsics=self._camera_intrinsics,
        cam_extrinsics=self._camera_extrinsics,
    )
```

### Observation Fields

| Field | Type | Description |
|-------|------|-------------|
| `proprio` | dict[str, list[float]] | Proprioceptive channels (see ProprioKeys) |
| `rgb` | dict[str, array] | Camera images by name |
| `depth` | dict[str, array] \| None | Optional depth images by camera name |
| `cam_intrinsics` | dict[str, list] | Camera intrinsics matrices by camera name |
| `cam_extrinsics` | dict[str, list] | Camera extrinsics matrices by camera name |
| `extra` | dict[str, Any] | Arbitrary additional data (e.g., task_prompt) |

**Common ProprioKeys:**
- `ProprioKeys.EE_POS` - End-effector position [x, y, z] meters
- `ProprioKeys.EE_QUAT` - End-effector quaternion [x, y, z, w]
- `ProprioKeys.EE_VEL_LIN` - Linear velocity [vx, vy, vz] m/s
- `ProprioKeys.EE_VEL_ANG` - Angular velocity [wx, wy, wz] rad/s
- `ProprioKeys.GRIPPER` - Gripper state [0.0-1.0]
- `ProprioKeys.BASE_POS` - Base position (for mobile robots)
- `ProprioKeys.BASE_HEADING` - Base heading (for navigation)

## Step 8: Handle Actions

Accept `Action` and convert to your simulator's format:

```python
from kinitro.rl_interface import Action, ActionKeys, coerce_action

def step(self, action: Action | dict | np.ndarray) -> tuple:
    """Execute action in environment."""

    # Coerce to Action (handles dict, array, or Action)
    action_obj = coerce_action(action)

    # Extract components using ActionKeys
    twist = action_obj.get_continuous(ActionKeys.EE_TWIST)  # [vx, vy, vz, wx, wy, wz]
    gripper = action_obj.get_continuous(ActionKeys.GRIPPER)  # [0.0-1.0]

    # Map to your simulator's action format
    # Example: convert normalized twist to velocity commands
    linear_vel = twist[:3] * self._max_linear_vel   # Scale to actual velocity
    angular_vel = twist[3:] * self._max_angular_vel

    # Execute in simulator
    self._simulator.set_ee_velocity(linear_vel, angular_vel)
    self._simulator.set_gripper(float(gripper[0]))
    self._simulator.step_physics()

    # ... rest of step logic
```

### Action Fields

| Field | Type | Description |
|-------|------|-------------|
| `continuous` | dict[str, list[float]] | Continuous control channels (see ActionKeys) |
| `discrete` | dict[str, int] | Discrete action selections |
| `extra` | dict[str, Any] | Arbitrary additional data |

**Common ActionKeys:**
- `ActionKeys.EE_TWIST` - Normalized twist [vx, vy, vz, wx, wy, wz] in [-1, 1]
- `ActionKeys.GRIPPER` - Gripper command [0.0-1.0]
- `ActionKeys.BASE_VEL` - Base velocity (for mobile robots)

## Step 9: Create the Container Actor

Create `environments/myrobot/env.py` with the `Actor` class:

```python
"""
Affinetes-compatible MyRobot evaluation environment.

Usage:
    import affinetes as af_env
    env = af_env.load_env(image="kinitro/myrobot:v1")
    result = await env.evaluate(task_id=123, base_url="https://...", env_id="myrobot/v0")
"""

import time
import traceback
import httpx
import numpy as np

from kinitro.environments import get_environment
from kinitro.environments.registry import get_all_environment_ids
from kinitro.rl_interface import Action, ActionKeys, coerce_action


class Actor:
    """MyRobot evaluation actor for affinetes."""

    def __init__(self):
        self._env_cache = {}

    def _get_env(self, env_id: str):
        if env_id not in self._env_cache:
            self._env_cache[env_id] = get_environment(env_id)
        return self._env_cache[env_id]

    async def list_environments(self) -> list[str]:
        """List available environments in this family."""
        return [e for e in get_all_environment_ids() if e.startswith("myrobot/")]

    async def evaluate(
        self,
        task_id: int,
        seed: int | None = None,
        base_url: str | None = None,
        env_id: str = "myrobot/v0",
        max_timesteps: int = 500,
        action_timeout: float = 0.5,
        use_images: bool = True,
        **kwargs,
    ) -> dict:
        """
        Evaluate a miner's policy.

        Returns:
            {
                "task_name": "robotics:myrobot/v0",
                "score": 0.0 or 1.0,
                "success": bool,
                "time_taken": float,
                "extra": {...},
                "error": optional error string
            }
        """
        if base_url is None:
            raise ValueError("base_url is required")

        seed = seed or task_id
        start_time = time.time()

        try:
            env = self._get_env(env_id)
            task_config = env.generate_task(seed=seed)

            # Reset miner policy
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{base_url}/reset",
                    json={"task_config": task_config.to_dict()}
                )

            # Reset environment
            obs = env.reset(task_config)

            # Evaluation loop
            for t in range(max_timesteps):
                # Get action from miner
                async with httpx.AsyncClient(timeout=action_timeout) as client:
                    resp = await client.post(
                        f"{base_url}/act",
                        json={"obs": obs.to_payload(include_images=use_images)}
                    )
                    action = resp.json().get("action")

                # Step environment
                obs, reward, done, info = env.step(action)

                if done:
                    break

            success = env.get_success()
            return {
                "task_name": f"robotics:{env_id}",
                "score": 1.0 if success else 0.0,
                "success": success,
                "time_taken": time.time() - start_time,
                "extra": {"task_id": task_id, "seed": seed},
            }

        except Exception as e:
            return {
                "task_name": f"robotics:{env_id}",
                "score": 0.0,
                "success": False,
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                "time_taken": time.time() - start_time,
                "extra": {"task_id": task_id, "seed": seed},
            }
```

## Step 10: Register the Environment

Add your environment to `kinitro/environments/registry.py`:

```python
def _make_myrobot_env(task: str) -> EnvFactory:
    """Create factory for MyRobot environment."""
    def factory() -> RoboticsEnvironment:
        from kinitro.environments.myrobot import MyRobotEnvironment
        return MyRobotEnvironment(task_name=task)
    return factory

ENVIRONMENTS: dict[str, EnvFactory] = {
    # ... existing environments ...

    # MyRobot manipulation environments
    "myrobot/pick-v0": _make_myrobot_env("pick-v0"),
    "myrobot/place-v0": _make_myrobot_env("place-v0"),
    "myrobot/slice-v0": _make_myrobot_env("slice-v0"),
}
```

## Step 11: Create the Dockerfile

```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY environments/myrobot/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy kinitro package (includes environments/ and rl_interface.py)
COPY kinitro /app/kinitro

# Copy environment actor
COPY environments/myrobot/env.py /app/env.py

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Affinetes looks for Actor class in env.py
CMD ["python", "-c", "from env import Actor; print('Actor loaded')"]
```

## Step 12: Build and Test

```bash
# Build the image
uv run kinitro env build myrobot --tag kinitro/myrobot:v1

# List environments
uv run kinitro env list

# Test locally (without miner)
uv run kinitro env test myrobot/pick-v0 --seed 42
```

---

# Complete Example: Robot Arm Manipulation Environment

Here's a complete example of a robot arm manipulation environment where tasks are procedurally generated.

## Scenario

- **Environment**: A tabletop with various objects (apples, plates, utensils, vegetables)
- **Tasks**: Procedurally generated manipulation tasks
- **Example task**: "Slice the apple and place the pieces on the plate"
- **Rewards**: Dense rewards for progress, sparse reward for completion
- **Success**: Task-specific completion criteria

## Directory Structure

```
environments/robomanip/
├── Dockerfile
├── env.py
├── requirements.txt
└── metadata.json

kinitro/environments/robomanip/
├── __init__.py
├── environment.py
├── task_types.py
├── task_generator.py
└── scene_generator.py
```

## task_types.py

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Any

class TaskType(Enum):
    PICK = "pick"
    PLACE = "place"
    SLICE = "slice"
    STACK = "stack"

@dataclass
class SceneObject:
    """Object in the manipulation scene."""
    object_id: str
    object_type: str
    position: dict[str, float]
    rotation: dict[str, float]
    pickupable: bool = False
    sliceable: bool = False
    receptacle: bool = False
    is_sliced: bool = False
    is_picked_up: bool = False
    parent_receptacles: list[str] = field(default_factory=list)

@dataclass
class TaskSpec:
    task_type: TaskType
    task_prompt: str
    target_object_id: str
    target_object_type: str
    destination_object_id: str | None = None
    destination_object_type: str | None = None
    initial_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type.value,
            "task_prompt": self.task_prompt,
            "target_object_id": self.target_object_id,
            "target_object_type": self.target_object_type,
            "destination_object_id": self.destination_object_id,
            "destination_object_type": self.destination_object_type,
            "initial_state": self.initial_state,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskSpec":
        return cls(
            task_type=TaskType(data["task_type"]),
            task_prompt=data["task_prompt"],
            target_object_id=data["target_object_id"],
            target_object_type=data["target_object_type"],
            destination_object_id=data.get("destination_object_id"),
            destination_object_type=data.get("destination_object_type"),
            initial_state=data.get("initial_state", {}),
        )
```

## task_generator.py

```python
import numpy as np
from .task_types import TaskType, TaskSpec, SceneObject

PROMPT_TEMPLATES = {
    TaskType.PICK: [
        "Pick up the {object}.",
        "Grab the {object} from the table.",
    ],
    TaskType.PLACE: [
        "Put the {object} on the {destination}.",
        "Place the {object} onto the {destination}.",
    ],
    TaskType.SLICE: [
        "Slice the {object} into pieces.",
        "Cut the {object} into a few pieces.",
        "Chop the {object}.",
    ],
}

class TaskGenerator:
    def __init__(self, task_types: list[TaskType] | None = None):
        self._task_types = task_types or list(TaskType)

    def generate_task(
        self,
        objects: list[SceneObject],
        rng: np.random.Generator,
    ) -> TaskSpec | None:
        """Generate a feasible task for the scene."""
        task_type = self._task_types[rng.integers(0, len(self._task_types))]

        if task_type == TaskType.PICK:
            return self._generate_pick_task(objects, rng)
        elif task_type == TaskType.PLACE:
            return self._generate_place_task(objects, rng)
        elif task_type == TaskType.SLICE:
            return self._generate_slice_task(objects, rng)
        return None

    def _generate_pick_task(self, objects: list[SceneObject], rng) -> TaskSpec | None:
        pickupables = [o for o in objects if o.pickupable and not o.is_picked_up]
        if not pickupables:
            return None
        target = pickupables[rng.integers(0, len(pickupables))]
        prompt = self._format_prompt(TaskType.PICK, target.object_type, None, rng)
        return TaskSpec(
            task_type=TaskType.PICK,
            task_prompt=prompt,
            target_object_id=target.object_id,
            target_object_type=target.object_type,
        )

    def _generate_place_task(self, objects: list[SceneObject], rng) -> TaskSpec | None:
        pickupables = [o for o in objects if o.pickupable]
        receptacles = [o for o in objects if o.receptacle]
        if not pickupables or not receptacles:
            return None
        target = pickupables[rng.integers(0, len(pickupables))]
        dest = receptacles[rng.integers(0, len(receptacles))]
        prompt = self._format_prompt(TaskType.PLACE, target.object_type, dest.object_type, rng)
        return TaskSpec(
            task_type=TaskType.PLACE,
            task_prompt=prompt,
            target_object_id=target.object_id,
            target_object_type=target.object_type,
            destination_object_id=dest.object_id,
            destination_object_type=dest.object_type,
        )

    def _generate_slice_task(self, objects: list[SceneObject], rng) -> TaskSpec | None:
        sliceables = [o for o in objects if o.sliceable and not o.is_sliced]
        if not sliceables:
            return None
        target = sliceables[rng.integers(0, len(sliceables))]
        prompt = self._format_prompt(TaskType.SLICE, target.object_type, None, rng)
        return TaskSpec(
            task_type=TaskType.SLICE,
            task_prompt=prompt,
            target_object_id=target.object_id,
            target_object_type=target.object_type,
        )

    def _format_prompt(self, task_type: TaskType, obj: str, dest: str | None, rng) -> str:
        templates = PROMPT_TEMPLATES[task_type]
        template = templates[rng.integers(0, len(templates))]
        obj_name = obj.lower().replace("_", " ")
        if dest:
            dest_name = dest.lower().replace("_", " ")
            return template.format(object=obj_name, destination=dest_name)
        return template.format(object=obj_name)
```

## environment.py

```python
"""Robot arm manipulation environment with procedural tasks."""

import numpy as np
from kinitro.environments.base import RoboticsEnvironment, TaskConfig
from kinitro.rl_interface import Observation, Action, ProprioKeys, ActionKeys, coerce_action
from .task_types import TaskType, TaskSpec, SceneObject
from .task_generator import TaskGenerator


class RoboManipEnvironment(RoboticsEnvironment):
    """
    Robot arm manipulation environment.

    Features:
    - Procedural scene generation (objects on table)
    - Multiple task types (pick, place, slice)
    - Natural language task instructions
    - Dense rewards for progress + sparse success signal
    """

    def __init__(
        self,
        task_name: str = "robomanip-v0",
        image_size: tuple[int, int] = (84, 84),
        max_episode_steps: int = 500,
        use_dense_rewards: bool = True,
    ):
        self._task_name = task_name
        self._image_size = image_size
        self._max_episode_steps = max_episode_steps
        self._use_dense_rewards = use_dense_rewards

        # Initialize simulator (placeholder - replace with actual simulator)
        self._simulator = None  # Your MuJoCo/PyBullet/Isaac sim

        # Task generator
        self._task_generator = TaskGenerator()

        # Episode state
        self._current_task: TaskSpec | None = None
        self._scene_objects: list[SceneObject] = []
        self._episode_steps = 0
        self._episode_success = False
        self._holding_object: str | None = None

    @property
    def env_name(self) -> str:
        return "robomanip"

    @property
    def task_name(self) -> str:
        return self._task_name

    def generate_task(self, seed: int) -> TaskConfig:
        """Generate procedural task from seed."""
        rng = np.random.default_rng(seed)

        # Generate random scene
        scene_objects = self._generate_scene(rng)

        # Generate task for this scene
        task_spec = self._task_generator.generate_task(scene_objects, rng)

        # Randomize physics
        physics_params = {
            "table_friction": float(rng.uniform(0.4, 0.8)),
            "object_mass_scale": float(rng.uniform(0.8, 1.2)),
            "gripper_force": float(rng.uniform(0.9, 1.1)),
        }

        return TaskConfig(
            env_name=self.env_name,
            task_name=self._task_name,
            seed=seed,
            physics_params=physics_params,
            domain_randomization={
                "scene_objects": [self._obj_to_dict(o) for o in scene_objects],
                "task_spec": task_spec.to_dict() if task_spec else None,
            },
        )

    def _generate_scene(self, rng: np.random.Generator) -> list[SceneObject]:
        """Generate random scene with objects on table."""
        objects = []

        # Object types and their properties
        object_defs = [
            ("Apple", True, True, False),      # pickupable, sliceable, receptacle
            ("Plate", False, False, True),
            ("Carrot", True, True, False),
            ("Knife", True, False, False),
            ("Bowl", False, False, True),
            ("Pencil", True, False, False),
        ]

        # Randomly select 3-6 objects
        n_objects = rng.integers(3, 7)
        selected = rng.choice(len(object_defs), size=n_objects, replace=False)

        for i, idx in enumerate(selected):
            name, pickupable, sliceable, receptacle = object_defs[idx]

            # Random position on table
            pos = {
                "x": float(rng.uniform(-0.3, 0.3)),
                "y": 0.05,  # On table surface
                "z": float(rng.uniform(-0.2, 0.2)),
            }

            objects.append(SceneObject(
                object_id=f"{name}_{i}",
                object_type=name,
                position=pos,
                rotation={"x": 0, "y": float(rng.uniform(0, 360)), "z": 0},
                pickupable=pickupable,
                sliceable=sliceable,
                receptacle=receptacle,
            ))

        return objects

    def _obj_to_dict(self, obj: SceneObject) -> dict:
        return {
            "object_id": obj.object_id,
            "object_type": obj.object_type,
            "position": obj.position,
            "rotation": obj.rotation,
            "pickupable": obj.pickupable,
            "sliceable": obj.sliceable,
            "receptacle": obj.receptacle,
        }

    def reset(self, task_config: TaskConfig) -> Observation:
        """Reset environment with task configuration."""
        self._episode_steps = 0
        self._episode_success = False
        self._holding_object = None

        # Load scene objects
        scene_data = task_config.domain_randomization.get("scene_objects", [])
        self._scene_objects = [
            SceneObject(**obj) for obj in scene_data
        ]

        # Load task spec
        task_data = task_config.domain_randomization.get("task_spec")
        self._current_task = TaskSpec.from_dict(task_data) if task_data else None

        # Apply physics params to simulator
        # self._simulator.set_physics(task_config.physics_params)

        # Reset simulator with scene
        # self._simulator.reset(self._scene_objects)

        return self._build_observation()

    def step(self, action: Action | dict | np.ndarray) -> tuple:
        """Execute action and return results."""
        self._episode_steps += 1
        action_obj = coerce_action(action)

        # Record state before action (for dense rewards)
        prev_distance = self._distance_to_goal()

        # Execute action in simulator
        # self._simulator.step(canonical_action)

        # Update scene state
        self._update_scene_state()

        # Check success
        self._check_success()

        # Compute reward
        reward = self._compute_reward(prev_distance)

        # Check termination
        done = self._episode_success or self._episode_steps >= self._max_episode_steps

        info = {
            "success": self._episode_success,
            "task_prompt": self._current_task.task_prompt if self._current_task else "",
            "episode_steps": self._episode_steps,
        }

        return self._build_observation(), reward, done, info

    def _compute_reward(self, prev_distance: float) -> float:
        """Compute reward signal."""
        if self._episode_success:
            return 1.0  # Success bonus

        if not self._use_dense_rewards:
            return 0.0  # Sparse: no reward until success

        # Dense reward: progress toward goal
        curr_distance = self._distance_to_goal()
        progress = prev_distance - curr_distance

        reward = progress * 5.0  # Scale progress reward

        # Grasp bonus
        if self._just_grasped_target():
            reward += 0.2

        return reward

    def _distance_to_goal(self) -> float:
        """Compute distance to task goal (for dense rewards)."""
        if self._current_task is None:
            return 0.0

        # Get end-effector position
        # ee_pos = self._simulator.get_ee_position()
        ee_pos = np.array([0, 0, 0])  # Placeholder

        # Get target object position
        target = self._get_object(self._current_task.target_object_id)
        if target is None:
            return 0.0

        target_pos = np.array([
            target.position["x"],
            target.position["y"],
            target.position["z"],
        ])

        return float(np.linalg.norm(ee_pos - target_pos))

    def _just_grasped_target(self) -> bool:
        """Check if we just grasped the target object."""
        # Implementation depends on simulator
        return False

    def _update_scene_state(self) -> None:
        """Update scene object states from simulator."""
        # Query simulator for current object states
        pass

    def _get_object(self, object_id: str) -> SceneObject | None:
        """Find object by ID."""
        for obj in self._scene_objects:
            if obj.object_id == object_id:
                return obj
        return None

    def _check_success(self) -> None:
        """Check if task is completed."""
        if self._current_task is None:
            self._episode_success = False
            return

        task = self._current_task

        if task.task_type == TaskType.PICK:
            self._episode_success = self._holding_object == task.target_object_id

        elif task.task_type == TaskType.PLACE:
            obj = self._get_object(task.target_object_id)
            if obj and task.destination_object_id:
                self._episode_success = (
                    task.destination_object_id in obj.parent_receptacles
                )

        elif task.task_type == TaskType.SLICE:
            obj = self._get_object(task.target_object_id)
            self._episode_success = obj is not None and obj.is_sliced

    def get_success(self) -> bool:
        """Return whether task was completed successfully."""
        return self._episode_success

    def _build_observation(self) -> Observation:
        """Build observation."""
        # Get state from simulator (placeholders)
        ee_pos = [0.0, 0.0, 0.0]
        ee_quat = [0.0, 0.0, 0.0, 1.0]
        ee_lin_vel = [0.0, 0.0, 0.0]
        ee_ang_vel = [0.0, 0.0, 0.0]
        gripper = 0.0 if self._holding_object is None else 1.0

        # Render camera (placeholder - 84x84 RGB)
        rgb_image = np.zeros((84, 84, 3), dtype=np.uint8)

        return Observation(
            proprio={
                ProprioKeys.EE_POS: ee_pos,
                ProprioKeys.EE_QUAT: ee_quat,
                ProprioKeys.EE_VEL_LIN: ee_lin_vel,
                ProprioKeys.EE_VEL_ANG: ee_ang_vel,
                ProprioKeys.GRIPPER: [gripper],
            },
            rgb={"ego": rgb_image},
            extra={
                "task_prompt": self._current_task.task_prompt if self._current_task else "",
            },
        )

    def close(self) -> None:
        """Clean up resources."""
        if self._simulator:
            # self._simulator.close()
            pass
```

## Summary Checklist

When adding a new environment, ensure you have:

- [ ] Created `environments/<family>/` directory with required files
- [ ] Implemented `RoboticsEnvironment` subclass with all abstract methods
- [ ] Defined `TaskConfig` generation with procedural parameters
- [ ] Specified task types and `TaskSpec` (for multi-task environments)
- [ ] Implemented reward signals (sparse for evaluation, optionally dense for training)
- [ ] Defined clear success criteria for each task type
- [ ] Created container `Actor` class with `evaluate()` method
- [ ] Registered environment(s) in `kinitro/environments/registry.py`
- [ ] Built and tested the Docker image

The key principle is that **evaluation scoring is always binary** (success/failure), determined by `get_success()`. Dense rewards can be used for training reference but don't affect the evaluation score.
