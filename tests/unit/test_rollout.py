"""Tests for rollout engine with vision-based observations."""

import asyncio
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from robo.evaluation.rollout import (
    ObservationData,
    RolloutConfig,
    decode_image_base64,
    encode_image_base64,
    run_episode,
)


class TestImageEncoding:
    """Tests for image encoding/decoding."""

    def test_encode_decode_roundtrip(self):
        """Image should be identical after encode/decode."""
        # Create a random RGB image
        original = np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)

        encoded = encode_image_base64(original)
        decoded = decode_image_base64(encoded)

        np.testing.assert_array_equal(decoded, original)

    def test_encoded_is_string(self):
        """Encoded image should be a base64 string."""
        image = np.zeros((84, 84, 3), dtype=np.uint8)
        encoded = encode_image_base64(image)

        assert isinstance(encoded, str)
        # Should be valid base64
        import base64

        base64.b64decode(encoded)  # Shouldn't raise


class TestObservationData:
    """Tests for ObservationData structure."""

    def test_to_dict(self):
        """ObservationData should convert to dict properly."""
        obs = ObservationData(
            end_effector_pos=[0.1, 0.2, 0.3],
            gripper_state=0.5,
            camera_images={"corner": "dGVzdA=="},  # base64 for "test"
        )

        d = obs.to_dict()

        assert d["end_effector_pos"] == [0.1, 0.2, 0.3]
        assert d["gripper_state"] == 0.5
        assert d["camera_images"]["corner"] == "dGVzdA=="


class MockEnvironment:
    """Mock environment for testing rollout."""

    def __init__(self):
        self._step_count = 0
        self._success = False
        self._last_obs = np.array([0.0, 0.5, 0.2, 1.0], dtype=np.float32)

    @property
    def action_shape(self):
        return (4,)

    @property
    def action_bounds(self):
        return (np.full(4, -1.0), np.full(4, 1.0))

    def reset(self, task_config):
        self._step_count = 0
        self._success = False
        return self._last_obs.copy()

    def step(self, action):
        self._step_count += 1
        reward = 1.0 if self._step_count > 10 else 0.1
        done = self._step_count >= 20
        if done:
            self._success = True
        return self._last_obs.copy(), reward, done, {}

    def get_success(self):
        return self._success

    def get_observation(self):
        """Return mock observation with camera views."""

        class MockObs:
            def __init__(self):
                self.camera_views = {
                    "corner": np.zeros((84, 84, 3), dtype=np.uint8),
                    "corner2": np.zeros((84, 84, 3), dtype=np.uint8),
                }

        return MockObs()

    def close(self):
        pass


class MockTaskConfig:
    """Mock task config."""

    def __init__(self):
        self.env_name = "mock"
        self.task_name = "test"
        self.seed = 42
        self.object_positions = np.zeros(3)
        self.target_positions = np.zeros(3)
        self.physics_params = {}
        self.domain_randomization = {}

    def to_dict(self):
        return {
            "env_name": self.env_name,
            "task_name": self.task_name,
            "seed": self.seed,
        }


class MockPolicy:
    """Mock policy that receives vision-based observations."""

    def __init__(self):
        self.reset_called = False
        self.last_observation = None
        self.act_count = 0

    async def reset(self, task_config):
        self.reset_called = True

    async def act(self, observation):
        self.last_observation = observation
        self.act_count += 1
        # Return valid action
        return [0.1, 0.2, 0.3, 0.4]


class TestRunEpisode:
    """Tests for run_episode function."""

    @pytest.mark.asyncio
    async def test_episode_runs_to_completion(self):
        """Episode should run and return result."""
        env = MockEnvironment()
        config = MockTaskConfig()
        policy = MockPolicy()

        result = await run_episode(
            env, config, policy, RolloutConfig(max_timesteps=50, include_cameras=False)
        )

        assert result.success is True
        assert result.timesteps == 20
        assert result.total_reward > 0

    @pytest.mark.asyncio
    async def test_policy_receives_dict_observation(self):
        """Policy should receive observation as dict with proper keys."""
        env = MockEnvironment()
        config = MockTaskConfig()
        policy = MockPolicy()

        await run_episode(
            env, config, policy, RolloutConfig(max_timesteps=5, include_cameras=False)
        )

        obs = policy.last_observation
        assert isinstance(obs, dict)
        assert "end_effector_pos" in obs
        assert "gripper_state" in obs
        assert "camera_images" in obs
        assert len(obs["end_effector_pos"]) == 3

    @pytest.mark.asyncio
    async def test_policy_receives_camera_images(self):
        """Policy should receive camera images when enabled."""
        env = MockEnvironment()
        config = MockTaskConfig()
        policy = MockPolicy()

        await run_episode(env, config, policy, RolloutConfig(max_timesteps=5, include_cameras=True))

        obs = policy.last_observation
        assert "camera_images" in obs
        assert len(obs["camera_images"]) > 0  # Has some camera images

        # Images should be base64 encoded
        for cam_name, img_b64 in obs["camera_images"].items():
            assert isinstance(img_b64, str)
            # Should decode without error
            decoded = decode_image_base64(img_b64)
            assert decoded.shape == (84, 84, 3)

    @pytest.mark.asyncio
    async def test_policy_reset_called(self):
        """Policy reset should be called before episode starts."""
        env = MockEnvironment()
        config = MockTaskConfig()
        policy = MockPolicy()

        await run_episode(
            env, config, policy, RolloutConfig(max_timesteps=5, include_cameras=False)
        )

        assert policy.reset_called is True


class TestRolloutConfig:
    """Tests for RolloutConfig."""

    def test_default_values(self):
        """Default config should have reasonable values."""
        config = RolloutConfig()

        assert config.max_timesteps == 500
        assert config.action_timeout_ms == 100
        assert config.include_cameras is True

    def test_custom_values(self):
        """Custom config values should be set."""
        config = RolloutConfig(
            max_timesteps=100,
            action_timeout_ms=50,
            include_cameras=False,
        )

        assert config.max_timesteps == 100
        assert config.action_timeout_ms == 50
        assert config.include_cameras is False
