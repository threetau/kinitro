# Kinitro Evaluation Environments

This directory contains the Affinetes-compatible evaluation environments for the Kinitro robotics subnet. Each environment family has its own directory with a Dockerfile, Actor class (`env.py`), and dependencies (`requirements.txt`).

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

# Build Genesis environment (~5GB image)
kinitro env build genesis --tag kinitro/genesis:v1

# Build and push to registry
kinitro env build metaworld --push --registry docker.io/myuser
```

## Environment Families

### MetaWorld

> **Note:** MetaWorld is included for local testing and development only. It is not actively used in mainnet evaluations.

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

### Genesis

Genesis physics simulation with a Unitree G1 humanoid robot in procedurally generated scenes.

**Supported environments:**

- `genesis/g1-v0` - Unitree G1 humanoid (43 actuated DOFs) with scene-grounded tasks:
  - NAVIGATE - Walk to a target object (success: within 0.5m)
  - PICKUP - Pick up a small object (success: lifted >0.15m)
  - PLACE - Pick up and place an object at a destination (success: within 0.3m)
  - PUSH - Push an object towards a destination (success: within 0.5m)

**Observations:**

- Proprioceptive (99 values): base position (3), base quaternion (4), base velocity (6), joint positions (43), joint velocities (43)
- Visual: 84x84 RGB ego camera mounted on robot torso

**Actions:** 43-dimensional continuous joint position targets in [-1, 1], scaled per-joint

**Scene generation:** 3-6 procedural objects (pickupable items + fixed landmarks) with randomized shapes, colors, sizes, and positions. Deterministic from seed.

**Control:** PD control at 50 Hz (2 physics substeps at 100 Hz per control step)

**Platform:** Linux recommended. GPU optional (CPU fallback via OSMesa for headless rendering).

**Image size:** ~5GB (includes Genesis engine, MuJoCo Menagerie, pre-compiled Taichi kernels)

**Rendering configuration (environment variables):**

The Genesis container reads these optional environment variables to tune rendering performance:

| Variable               | Default | Description                                   |
| ---------------------- | ------- | --------------------------------------------- |
| `GENESIS_RENDER_DEPTH` | `0`     | Set to `1` to enable depth rendering (slower) |

Example: enable depth rendering if your policy uses depth images:

```bash
GENESIS_RENDER_DEPTH=1
```

## Backend Configuration

Configure your executor to use the appropriate image for each environment family. Set the environment variable:

```bash
KINITRO_EXECUTOR_EVAL_IMAGES='{"metaworld": "kinitro/metaworld:v1", "genesis": "kinitro/genesis:v1"}'
```

---

## Adding New Environments

Copy the `_template/` directory and customize:

```bash
cp -r environments/_template environments/myenv
```

### Required Files

| File               | Purpose                              |
| ------------------ | ------------------------------------ |
| `metadata.json`    | Display name and description         |
| `requirements.txt` | Python dependencies                  |
| `Dockerfile`       | Container build spec                 |
| `env.py`           | Actor class with `evaluate()` method |

### Key Concepts

- **Actor.evaluate()**: Main entry point called by Affinetes
- **TaskConfig**: Procedural task parameters (seed, positions, physics)
- **Observation**: Proprio dict + camera images (see `kinitro/rl_interface.py`)
- **Action**: Continuous/discrete channels (see `kinitro/rl_interface.py`)

### Implementation Steps

1. Add your simulator to `requirements.txt`
2. Update Dockerfile system deps if needed
3. Implement `env.py` Actor class (follow template TODOs)
4. Register environment in `kinitro/environments/registry.py`
5. Build and test: `kinitro env build myenv --tag myenv:v1`

For reference implementations, see `metaworld/` and `genesis/`.
