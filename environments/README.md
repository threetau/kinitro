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

## Adding New Environments

Copy the `_template/` directory and customize:

```bash
cp -r environments/_template environments/myenv
```

### Required Files

| File | Purpose |
|------|---------|
| `metadata.json` | Display name and description |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container build spec |
| `env.py` | Actor class with `evaluate()` method |

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

For reference implementations, see `metaworld/` and `procthor/`.
