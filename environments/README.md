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

Use the `kinitro build-env` command to build environment-specific Docker images:

```bash
# Build MetaWorld environment (~1GB image)
kinitro build-env metaworld --tag kinitro/metaworld:v1

# Build ProcTHOR environment (~3GB image)
kinitro build-env procthor --tag kinitro/procthor:v1

# Build and push to registry
kinitro build-env metaworld --push --registry docker.io/myuser
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

## Adding New Environment Families

To add a new environment family:

1. Create a new directory under `environments/` (e.g., `environments/myenv/`)
2. Add required files:
   - `Dockerfile` - Container build instructions
   - `env.py` - Actor class with `evaluate()` method
   - `requirements.txt` - Python dependencies
   - `metadata.json` - Family display info: `{"name": "MYENV", "description": "My Environment"}`
3. Register environments in `kinitro/environments/registry.py`:
   - Add entries to `ENVIRONMENTS` dict with your environment IDs
4. Build with `kinitro build-env myenv --tag kinitro/myenv:v1`

The CLI and list commands will automatically discover the new family from `metadata.json`.
