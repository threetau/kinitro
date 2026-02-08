"""
Pre-warm Taichi kernel cache for Genesis physics simulation.

Run during `docker build` to bake compiled kernels into the image layer.
This avoids ~36s of JIT compilation on every fresh container startup.

The script builds a minimal headless scene matching the runtime physics
structure (ground plane, G1 robot, Box/Sphere/Cylinder primitives) to
trigger all physics kernel compilation.

Genesis internally creates a visualizer even when renderer=None, which
fails without EGL during docker build. The physics kernels are compiled
before the visualizer step, so we catch the EGL error and exit cleanly.
Genesis's atexit handler caches the compiled kernels regardless.
"""

import os
import time

import genesis as gs

MENAGERIE_ROBOT = "unitree_g1/g1_with_hands.xml"


def warmup() -> None:
    gs.init(backend=getattr(gs, "cpu"))

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
        renderer=None,
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

    t0 = time.perf_counter()
    try:
        scene.build(n_envs=1)
    except Exception as e:
        # Genesis compiles physics kernels before building the visualizer.
        # Without EGL (typical in docker build), the visualizer init fails,
        # but the physics kernels are already compiled and will be cached
        # by Genesis's atexit handler.
        build_elapsed = time.perf_counter() - t0
        print(f"[warmup] scene.build() physics kernels compiled in {build_elapsed:.1f}s")
        print(f"[warmup] Visualizer init failed (expected without EGL): {type(e).__name__}")
        print("[warmup] Kernel cache warm-up complete (cached on process exit)")
        return

    build_elapsed = time.perf_counter() - t0
    print(f"[warmup] scene.build() completed in {build_elapsed:.1f}s")

    t1 = time.perf_counter()
    scene.step()
    step_elapsed = time.perf_counter() - t1
    print(f"[warmup] scene.step()  completed in {step_elapsed:.1f}s")

    print(f"[warmup] Kernel cache warm-up complete in {build_elapsed + step_elapsed:.1f}s")


if __name__ == "__main__":
    warmup()
