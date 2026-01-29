"""E2E tests for MetaWorld action format mappings."""

import os
import platform

import numpy as np
import pytest
from structlog.testing import capture_logs

from kinitro.environments.base import TaskConfig
from kinitro.environments.metaworld_env import MetaWorldEnvironment
from kinitro.rl_interface import CanonicalAction


def _ensure_cpu_rendering() -> None:
    if os.environ.get("MUJOCO_GL"):
        return
    if platform.system() == "Darwin":
        os.environ["MUJOCO_GL"] = "cgl"
    else:
        os.environ["MUJOCO_GL"] = "osmesa"


@pytest.mark.integration
@pytest.mark.metaworld
class TestMetaWorldActionFormatE2E:
    def setup_method(self) -> None:
        _ensure_cpu_rendering()

        # Hard-fail if dependencies are missing.

    def _make_env(self, action_format: str = "auto") -> MetaWorldEnvironment:
        return MetaWorldEnvironment(
            "pick-place-v3",
            use_camera=True,
            image_size=(84, 84),
            action_format=action_format,
            warn_on_orientation_mismatch=True,
        )

    def _task_config(self, seed: int) -> TaskConfig:
        return TaskConfig(env_name="metaworld", task_name="pick-place-v3", seed=seed)

    def test_auto_xyz_gripper_action(self, capsys: pytest.CaptureFixture[str]) -> None:
        env = self._make_env(action_format="auto")
        try:
            task_config = self._task_config(seed=123)
            _ = env.reset(task_config)

            assert env.action_shape == (4,)

            action = CanonicalAction(
                twist_ee_norm=[0.2, -0.1, 0.05, 0.4, -0.2, 0.1],
                gripper_01=1.0,
            )

            original_step = env._env.step
            captured: dict[str, np.ndarray] = {}

            def _wrapped_step(native_action):
                captured["action"] = np.array(native_action)
                return original_step(native_action)

            env._env.step = _wrapped_step

            try:
                with capture_logs() as logs:
                    obs, reward, done, info = env.step(action)
            finally:
                env._env.step = original_step

            assert captured["action"].shape == (4,)
            assert np.all(captured["action"][:3] != 0.0)

            assert isinstance(obs.rgb, dict)
            if obs.rgb:
                for image in obs.rgb.values():
                    arr = np.array(image)
                    assert arr.shape == (84, 84, 3)
                    assert arr.dtype.kind in {"i", "u"}
                    assert arr.min() >= 0
                    assert arr.max() <= 255
            else:
                camera_warnings = [
                    entry
                    for entry in logs
                    if "camera" in entry.get("event", "") or "camera" in entry.get("message", "")
                ]
                if not camera_warnings:
                    captured_out = capsys.readouterr().out
                    assert "cameras unavailable" in captured_out

            assert isinstance(reward, float)
            assert isinstance(done, bool)
            assert isinstance(info, dict)

            warnings = [
                entry
                for entry in logs
                if "angular twist" in entry.get("event", "")
                or "angular twist" in entry.get("message", "")
            ]
            assert warnings

        finally:
            env.close()

    def test_warn_once_for_angular_twist(self) -> None:
        env = self._make_env(action_format="auto")
        try:
            task_config = self._task_config(seed=321)
            _ = env.reset(task_config)

            action = CanonicalAction(
                twist_ee_norm=[0.0, 0.0, 0.0, 0.3, 0.2, 0.1],
                gripper_01=0.0,
            )

            with capture_logs() as logs:
                for _ in range(3):
                    env.step(action)

            warnings = [
                entry
                for entry in logs
                if "angular twist" in entry.get("event", "")
                or "angular twist" in entry.get("message", "")
            ]
            assert len(warnings) == 1

        finally:
            env.close()

    def test_action_format_mismatch_raises(self) -> None:
        env = self._make_env(action_format="xyz_quat")
        try:
            task_config = self._task_config(seed=999)

            with pytest.raises(ValueError, match="action_format mismatch"):
                env.reset(task_config)
        finally:
            env.close()
