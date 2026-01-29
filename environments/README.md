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
# Build MetaWorld environment (~400MB image)
kinitro build-env metaworld --tag kinitro/metaworld:v1

# Build ProcTHOR environment (~1.5GB image)
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

**Image size:** ~400MB

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

**Image size:** ~1.5GB

## Backend Configuration

Configure your backend to use the appropriate image for each environment family:

```yaml
eval_images:
  metaworld: "kinitro/metaworld:v1"
  procthor: "kinitro/procthor:v1"
```

Or use a single image tag that gets automatically selected based on the `env_id` prefix.

## Migration from eval-env/

The old `eval-env/` directory contained a monolithic image with all environments. This has been deprecated in favor of the per-environment structure for:

1. **Smaller images** - Only include dependencies you need
2. **Faster builds** - Build only what you need
3. **Better platform compatibility** - MetaWorld works everywhere, ProcTHOR is x86_64 Linux only
4. **Cleaner separation** - Each environment is self-contained

To migrate:
1. Replace `kinitro build-eval-env` with `kinitro build-env <family>`
2. Update your backend config to use per-family images
3. The API remains the same - just the Docker images are different
