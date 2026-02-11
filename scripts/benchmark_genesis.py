"""Benchmark Genesis environment to identify performance bottlenecks.

Profiles each component of the step loop independently:
  1. Physics stepping (scene.step)
  2. Robot state reads (GPU->CPU tensor transfers)
  3. Object state reads
  4. Camera rendering (move_to_attach + render)
  5. Image encoding (numpy -> base64)
  6. Observation building (full _build_observation)
  7. Action application (control_dofs_position)
  8. Full step() call (end-to-end)

Usage:
    MUJOCO_GL=egl uv run python scripts/benchmark_genesis.py
    MUJOCO_GL=egl uv run python scripts/benchmark_genesis.py --steps 200
    MUJOCO_GL=egl uv run python scripts/benchmark_genesis.py --no-render
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TimingBucket:
    name: str
    times: list[float] = field(default_factory=list)

    def record(self, dt: float) -> None:
        self.times.append(dt)

    @property
    def total(self) -> float:
        return sum(self.times)

    @property
    def mean(self) -> float:
        return self.total / len(self.times) if self.times else 0.0

    @property
    def median(self) -> float:
        if not self.times:
            return 0.0
        s = sorted(self.times)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    @property
    def p95(self) -> float:
        if not self.times:
            return 0.0
        s = sorted(self.times)
        return s[int(len(s) * 0.95)]

    @property
    def count(self) -> int:
        return len(self.times)


def benchmark(num_steps: int = 100, skip_render: bool = False) -> None:
    # ── Genesis init ─────────────────────────────────────────────
    print("=" * 70)
    print("Genesis Environment Benchmark")
    print("=" * 70)

    t0 = time.perf_counter()
    from kinitro.environments.genesis.base import _init_genesis  # noqa: PLC0415

    _init_genesis()
    import genesis as gs  # noqa: PLC0415

    t_init = time.perf_counter() - t0
    print(f"\n[init] Genesis initialized in {t_init:.2f}s")
    backend_name = getattr(gs, "_backend", None) or os.environ.get("GENESIS_BACKEND", "unknown")
    print(f"       Backend: {backend_name}")

    # ── Environment setup ────────────────────────────────────────
    from kinitro.environments.genesis.envs.g1_humanoid import G1Environment  # noqa: PLC0415

    env = G1Environment()
    task_config = env.generate_task(seed=42)

    t0 = time.perf_counter()
    obs = env.reset(task_config)
    t_reset = time.perf_counter() - t0
    print(f"[reset] Scene built + first obs in {t_reset:.2f}s")

    robot = env._robot
    scene = env._scene
    camera = env._camera
    n_dofs = env._robot_config.num_actuated_dofs
    n_objects = len(env._object_entities)

    print("\n--- Config ---")
    print(f"  Robot DOFs:     {n_dofs}")
    print(f"  Scene objects:  {n_objects}")
    print(f"  Image size:     {env.IMAGE_SIZE}x{env.IMAGE_SIZE}")
    print(f"  SIM_DT:         {env.SIM_DT}s  ({int(1 / env.SIM_DT)} Hz physics)")
    print(f"  CONTROL_DT:     {env.CONTROL_DT}s ({int(1 / env.CONTROL_DT)} Hz control)")
    print(f"  Substeps/ctrl:  {int(env.CONTROL_DT / env.SIM_DT)}")
    print(f"  Benchmark steps: {num_steps}")
    print(f"  Rendering:      {'DISABLED' if skip_render else 'ENABLED'}")

    # ── Timing buckets ───────────────────────────────────────────
    t_physics_1 = TimingBucket("physics_1_substep")
    t_physics_2 = TimingBucket("physics_2_substeps")
    t_robot_state = TimingBucket("robot_state_read")
    t_robot_state_parts = {
        "get_pos": TimingBucket("  get_pos"),
        "get_quat": TimingBucket("  get_quat"),
        "get_vel": TimingBucket("  get_vel"),
        "get_ang": TimingBucket("  get_ang"),
        "get_dofs_position": TimingBucket("  get_dofs_position"),
        "get_dofs_velocity": TimingBucket("  get_dofs_velocity"),
    }
    t_object_state = TimingBucket("object_state_read")
    t_camera_move = TimingBucket("camera_move_to_attach")
    t_camera_render = TimingBucket("camera_render")
    t_camera_total = TimingBucket("camera_total")
    t_img_encode = TimingBucket("image_encode")
    t_obs_build = TimingBucket("obs_build_full")
    t_action_apply = TimingBucket("action_apply")
    t_reward_check = TimingBucket("reward+success_check")
    t_full_step = TimingBucket("full_step_e2e")

    # Create a dummy action
    dummy_action_data = np.zeros(n_dofs, dtype=np.float32)
    from kinitro.rl_interface import Action, ActionKeys  # noqa: PLC0415

    dummy_action = Action(continuous={ActionKeys.JOINT_POS_TARGET: dummy_action_data.tolist()})

    # Warmup: 5 steps to stabilize JIT/caches
    print("\nWarming up (5 steps)...")
    for _ in range(5):
        env.step(dummy_action)

    # ── Main benchmark loop ──────────────────────────────────────
    print(f"Benchmarking {num_steps} steps...\n")
    actuated_dof_idx = list(range(6, 6 + n_dofs))

    for i in range(num_steps):
        step_t0 = time.perf_counter()

        # 1. Action apply
        t = time.perf_counter()
        target_pos = env._default_dof_pos + dummy_action_data * env._action_scale
        robot.control_dofs_position(target_pos, dofs_idx_local=actuated_dof_idx)
        t_action_apply.record(time.perf_counter() - t)

        # 2. Physics - single substep
        t = time.perf_counter()
        scene.step()
        t_physics_1.record(time.perf_counter() - t)

        # 3. Physics - second substep (to match CONTROL_DT)
        t = time.perf_counter()
        scene.step()
        dt1 = time.perf_counter() - t
        t_physics_2.record(t_physics_1.times[-1] + dt1)

        # 4. Robot state read (decomposed)
        t = time.perf_counter()
        t_sub = time.perf_counter()
        pos = robot.get_pos().cpu().numpy().flatten()
        t_robot_state_parts["get_pos"].record(time.perf_counter() - t_sub)

        t_sub = time.perf_counter()
        quat = robot.get_quat().cpu().numpy().flatten()
        t_robot_state_parts["get_quat"].record(time.perf_counter() - t_sub)

        t_sub = time.perf_counter()
        vel = robot.get_vel().cpu().numpy().flatten()
        t_robot_state_parts["get_vel"].record(time.perf_counter() - t_sub)

        t_sub = time.perf_counter()
        ang = robot.get_ang().cpu().numpy().flatten()
        t_robot_state_parts["get_ang"].record(time.perf_counter() - t_sub)

        t_sub = time.perf_counter()
        dof_pos = robot.get_dofs_position().cpu().numpy().flatten()
        t_robot_state_parts["get_dofs_position"].record(time.perf_counter() - t_sub)

        t_sub = time.perf_counter()
        dof_vel = robot.get_dofs_velocity().cpu().numpy().flatten()
        t_robot_state_parts["get_dofs_velocity"].record(time.perf_counter() - t_sub)

        t_robot_state.record(time.perf_counter() - t)

        # 5. Object state read
        t = time.perf_counter()
        for entity in env._object_entities:
            entity.get_pos().cpu().numpy().flatten()
        t_object_state.record(time.perf_counter() - t)

        # 6. Camera rendering
        if not skip_render and camera is not None:
            t = time.perf_counter()
            if camera._attached_link is not None:
                t_move = time.perf_counter()
                camera.move_to_attach()
                t_camera_move.record(time.perf_counter() - t_move)

            t_render = time.perf_counter()
            rgb, depth, _seg, _normal = camera.render(rgb=True, depth=True)
            t_camera_render.record(time.perf_counter() - t_render)
            t_camera_total.record(time.perf_counter() - t)

            # 7. Image encoding
            t = time.perf_counter()
            from kinitro.rl_interface import encode_image  # noqa: PLC0415

            if rgb is not None:
                if rgb.dtype != np.uint8:
                    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                encode_image(rgb)
            if depth is not None:
                encode_image(depth)
            t_img_encode.record(time.perf_counter() - t)

        # 8. Reward/success check (lightweight but measure anyway)
        t = time.perf_counter()
        robot_state = {
            "base_pos": pos,
            "base_quat": quat,
            "base_vel": vel,
            "base_ang_vel": ang,
            "dof_pos": dof_pos[6 : 6 + n_dofs],
            "dof_vel": dof_vel[6 : 6 + n_dofs],
        }
        object_states = {}
        for j, entity in enumerate(env._object_entities):
            if j < len(env._scene_objects):
                object_states[env._scene_objects[j].object_id] = pos
        if env._current_task:
            env._compute_reward(robot_state, object_states, env._current_task)
            env._check_success(robot_state, object_states, env._current_task)
        t_reward_check.record(time.perf_counter() - t)

        t_full_step.record(time.perf_counter() - step_t0)

    # ── Now run full step() for comparison ───────────────────────
    t_full_step_method = TimingBucket("env.step()_method")
    for _ in range(num_steps):
        t = time.perf_counter()
        obs, reward, done, info = env.step(dummy_action)
        t_full_step_method.record(time.perf_counter() - t)
        if done:
            env.reset(task_config)

    # ── Results ──────────────────────────────────────────────────
    all_buckets = [
        t_full_step,
        t_full_step_method,
        None,  # separator
        t_physics_1,
        t_physics_2,
        t_action_apply,
        None,
        t_robot_state,
        *t_robot_state_parts.values(),
        t_object_state,
        None,
        t_camera_total,
        t_camera_move,
        t_camera_render,
        t_img_encode,
        None,
        t_obs_build,
        t_reward_check,
    ]

    fps_decomposed = 1.0 / t_full_step.mean if t_full_step.mean > 0 else 0
    fps_method = 1.0 / t_full_step_method.mean if t_full_step_method.mean > 0 else 0

    print("=" * 70)
    print(f"RESULTS  ({num_steps} steps, rendering {'OFF' if skip_render else 'ON'})")
    print("=" * 70)
    print(f"{'Component':<30} {'Mean':>10} {'Median':>10} {'P95':>10} {'% of step':>10}")
    print("-" * 70)

    step_mean = t_full_step.mean
    for b in all_buckets:
        if b is None:
            print("-" * 70)
            continue
        if not b.times:
            continue
        pct = (b.mean / step_mean * 100) if step_mean > 0 else 0
        print(
            f"{b.name:<30} {b.mean * 1000:>8.2f}ms {b.median * 1000:>8.2f}ms "
            f"{b.p95 * 1000:>8.2f}ms {pct:>8.1f}%"
        )

    print("-" * 70)
    print(f"\nEffective FPS (decomposed):  {fps_decomposed:.2f}")
    print(f"Effective FPS (env.step()):   {fps_method:.2f}")
    print("Step time budget @ 50Hz:      20.00ms")
    print(f"Actual step time:             {t_full_step.mean * 1000:.2f}ms")
    print(f"Slowdown vs realtime:         {t_full_step.mean / env.CONTROL_DT:.1f}x")

    # Pie chart breakdown
    print("\n--- Time Budget Breakdown (per step) ---")
    components = [
        ("Physics (2 substeps)", t_physics_2.mean),
        ("Robot state read", t_robot_state.mean),
        ("Object state read", t_object_state.mean),
        ("Camera total", t_camera_total.mean if t_camera_total.times else 0),
        ("Image encode", t_img_encode.mean if t_img_encode.times else 0),
        ("Action apply", t_action_apply.mean),
        ("Reward/success", t_reward_check.mean),
    ]
    accounted = sum(v for _, v in components)
    unaccounted = step_mean - accounted
    components.append(("Overhead/other", max(0, unaccounted)))

    for name, val in sorted(components, key=lambda x: -x[1]):
        bar_len = int(val / step_mean * 40) if step_mean > 0 else 0
        pct = val / step_mean * 100 if step_mean > 0 else 0
        print(f"  {name:<22} {val * 1000:>7.2f}ms  {pct:>5.1f}%  {'█' * bar_len}")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Genesis environment")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to benchmark")
    parser.add_argument("--no-render", action="store_true", help="Disable camera rendering")
    args = parser.parse_args()
    benchmark(num_steps=args.steps, skip_render=args.no_render)
