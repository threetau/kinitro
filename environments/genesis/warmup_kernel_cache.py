"""
Pre-warm Taichi kernel cache for Genesis physics simulation.

Run during `docker build` to bake compiled kernels into the image layer.
This avoids ~36s of JIT compilation on every fresh container startup.

The script builds a minimal headless scene matching the runtime physics
structure (ground plane, G1 robot, Box/Sphere/Cylinder primitives) to
trigger all physics and rendering kernel compilation.

Uses OSMesa (CPU software rendering) because docker build has no GPU.
With OSMesa the full scene.build() succeeds — including the rasterizer —
so both physics and rendering kernels get cached.
"""

import os
import time

# Must be set BEFORE importing genesis — PyOpenGL locks the platform backend
# on first import. No GPU during docker build, so force OSMesa (CPU software rendering).
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import genesis as gs  # noqa: E402

# Patch OSMesa framebuffer support — Genesis hardcodes supports_framebuffers=False
# but modern OSMesa (GL 3.3 core via libosmesa6-dev) fully supports FBOs.
# Without this patch, camera.render() hits a broken non-FBO code path.
from genesis.ext.pyrender.platforms.osmesa import OSMesaPlatform  # noqa: E402

OSMesaPlatform.supports_framebuffers = lambda self: True  # type: ignore[assignment]

MENAGERIE_ROBOT = "unitree_g1/g1_with_hands.xml"
IMAGE_SIZE = 84  # Small image size to minimize OSMesa rendering time during warm-up, while still triggering all relevant kernels.


def warmup() -> None:
    gs.init(backend=getattr(gs, "cpu"))

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
        renderer=gs.renderers.Rasterizer(),
    )

    # Ground plane
    scene.add_entity(gs.morphs.Plane())

    # G1 humanoid robot (matches runtime spawn in base.py)
    menagerie_path = os.environ.get("GENESIS_MENAGERIE_PATH", "/opt/menagerie")
    mjcf_path = os.path.join(menagerie_path, MENAGERIE_ROBOT)
    scene.add_entity(
        gs.morphs.MJCF(
            file=mjcf_path,
            pos=(0.0, 0.0, 0.75),
            quat=(1.0, 0.0, 0.0, 0.0),
        )
    )

    # Primitive morphs used in scene_generator.py — both fixed and dynamic
    # variants so Taichi compiles kernels for each combination.
    for fixed in (True, False):
        scene.add_entity(
            gs.morphs.Box(
                pos=(1.0, 0.0, 0.05),
                size=(0.05, 0.05, 0.05),
                fixed=fixed,
            )
        )
        scene.add_entity(
            gs.morphs.Sphere(
                pos=(0.0, 1.0, 0.05),
                radius=0.05,
                fixed=fixed,
            )
        )
        scene.add_entity(
            gs.morphs.Cylinder(
                pos=(-1.0, 0.0, 0.05),
                radius=0.05,
                height=0.10,
                fixed=fixed,
            )
        )

    # Camera for rendering validation
    camera = scene.add_camera(
        res=(IMAGE_SIZE, IMAGE_SIZE),
        pos=(0.0, -2.0, 1.5),
        lookat=(0.0, 0.0, 0.75),
        fov=90,
    )

    t0 = time.perf_counter()
    scene.build(n_envs=1)
    build_elapsed = time.perf_counter() - t0
    print(f"[warmup] scene.build() completed in {build_elapsed:.1f}s")

    t1 = time.perf_counter()
    scene.step()
    step_elapsed = time.perf_counter() - t1
    print(f"[warmup] scene.step()  completed in {step_elapsed:.1f}s")

    # Validate camera rendering works under OSMesa.
    # This is intentionally fail-fast: if camera.render() fails, the Docker
    # build should fail so broken OSMesa configs are caught at build time.
    rgb, depth, _seg, _normal = camera.render(rgb=True, depth=True)
    if rgb is None:
        raise RuntimeError(
            "Camera render validation failed: rgb is None. "
            "OSMesa may not be configured correctly (is libosmesa6-dev installed?)."
        )
    if rgb.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(
            f"Camera render validation failed: expected rgb spatial dims "
            f"({IMAGE_SIZE}, {IMAGE_SIZE}), got {rgb.shape[:2]}."
        )
    print(f"[warmup] camera.render() completed — rgb shape={rgb.shape}")

    print(f"[warmup] Kernel cache warm-up complete in {build_elapsed + step_elapsed:.1f}s")


if __name__ == "__main__":
    warmup()
