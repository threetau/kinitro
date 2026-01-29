# Kinitro Evaluation Environments

This directory contains the affinetes-compatible evaluation environments for the Kinitro robotics subnet. Each environment family has its own directory with a Dockerfile, Actor class (`env.py`), and dependencies (`requirements.txt`).

## Directory Structure

```
environments/
├── metaworld/          # MuJoCo-based manipulation tasks
│   ├── Dockerfile
│   ├── env.py          # Actor class for metaworld/* environments
│   └── requirements.txt
│
└── procthor/           # AI2-THOR procedural house tasks
    ├── Dockerfile
    ├── env.py          # Actor class for procthor/* environments
    └── requirements.txt
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

Configure your backend to use the appropriate image for each environment family:

```yaml
eval_images:
  metaworld: "kinitro/metaworld:v1"
  procthor: "kinitro/procthor:v1"
```

Or use a single image tag that gets automatically selected based on the `env_id` prefix.

## Adding New Environment Families

To add a new environment family:

1. Create a new directory under `environments/` (e.g., `environments/myenv/`)
2. Add `Dockerfile`, `env.py` (Actor class), and `requirements.txt`
3. Register environments in `kinitro/environments/registry.py`:
   - Add entries to `ENVIRONMENTS` dict
   - Add metadata to `FAMILY_METADATA` dict
4. Build with `kinitro build-env myenv --tag kinitro/myenv:v1`

The CLI and list commands will automatically pick up the new family.
