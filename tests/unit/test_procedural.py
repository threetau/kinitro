"""Tests for procedural generation."""

import numpy as np
import pytest

from robo.environments.procedural import (
    ProceduralTaskGenerator,
    generate_seed_from_block,
    randomize_physics,
    randomize_positions,
)


class TestRandomizePositions:
    """Tests for position randomization."""

    def test_output_shape(self):
        """Output should be same shape as base."""
        rng = np.random.default_rng(42)
        base = np.array([1.0, 2.0, 3.0])
        result = randomize_positions(base, rng, [0.1, 0.1, 0.1])

        assert result.shape == base.shape

    def test_within_range(self):
        """Positions should stay within specified range."""
        rng = np.random.default_rng(42)
        base = np.array([0.0, 0.0, 0.0])
        range_xyz = np.array([0.1, 0.2, 0.3])

        for _ in range(100):
            result = randomize_positions(base, rng, range_xyz)
            assert np.all(np.abs(result - base) <= range_xyz)

    def test_deterministic_with_seed(self):
        """Same seed should give same result."""
        base = np.array([1.0, 1.0, 1.0])

        rng1 = np.random.default_rng(42)
        result1 = randomize_positions(base, rng1, [0.1, 0.1, 0.1])

        rng2 = np.random.default_rng(42)
        result2 = randomize_positions(base, rng2, [0.1, 0.1, 0.1])

        np.testing.assert_array_equal(result1, result2)


class TestRandomizePhysics:
    """Tests for physics randomization."""

    def test_all_params_generated(self):
        """All requested params should be in result."""
        rng = np.random.default_rng(42)
        params = {
            "friction": (0.5, 1.5),
            "damping": (0.8, 1.2),
            "mass": (0.9, 1.1),
        }

        result = randomize_physics(rng, params)

        assert set(result.keys()) == {"friction", "damping", "mass"}

    def test_within_range(self):
        """Values should be within specified range."""
        rng = np.random.default_rng(42)
        params = {"friction": (0.5, 1.5)}

        for _ in range(100):
            result = randomize_physics(rng, params)
            assert 0.5 <= result["friction"] <= 1.5


class TestGenerateSeedFromBlock:
    """Tests for deterministic seed generation."""

    def test_deterministic(self):
        """Same inputs should give same seed."""
        seed1 = generate_seed_from_block(1000, "env_a", "hotkey123", 0)
        seed2 = generate_seed_from_block(1000, "env_a", "hotkey123", 0)

        assert seed1 == seed2

    def test_different_blocks_different_seeds(self):
        """Different blocks should give different seeds."""
        seed1 = generate_seed_from_block(1000, "env_a", "hotkey123", 0)
        seed2 = generate_seed_from_block(1001, "env_a", "hotkey123", 0)

        assert seed1 != seed2

    def test_different_envs_different_seeds(self):
        """Different environments should give different seeds."""
        seed1 = generate_seed_from_block(1000, "env_a", "hotkey123", 0)
        seed2 = generate_seed_from_block(1000, "env_b", "hotkey123", 0)

        assert seed1 != seed2

    def test_different_validators_different_seeds(self):
        """Different validators should get different seeds."""
        seed1 = generate_seed_from_block(1000, "env_a", "validator1", 0)
        seed2 = generate_seed_from_block(1000, "env_a", "validator2", 0)

        assert seed1 != seed2

    def test_positive_32bit(self):
        """Seed should be positive 32-bit integer."""
        for i in range(100):
            seed = generate_seed_from_block(i, f"env_{i}", "hotkey", i)
            assert 0 <= seed <= 0x7FFFFFFF


class TestProceduralTaskGenerator:
    """Tests for the procedural task generator."""

    def test_generate_returns_dict(self):
        """Generate should return dict with expected keys."""
        gen = ProceduralTaskGenerator(env_id="test_env")
        result = gen.generate(seed=42)

        assert "object_positions" in result
        assert "target_positions" in result
        assert "physics_params" in result
        assert "domain_randomization" in result

    def test_deterministic(self):
        """Same seed should give same result."""
        gen = ProceduralTaskGenerator(env_id="test_env")

        result1 = gen.generate(seed=42)
        result2 = gen.generate(seed=42)

        np.testing.assert_array_equal(result1["object_positions"], result2["object_positions"])
        assert result1["physics_params"] == result2["physics_params"]

    def test_different_seeds_different_results(self):
        """Different seeds should give different results."""
        gen = ProceduralTaskGenerator(env_id="test_env")

        result1 = gen.generate(seed=42)
        result2 = gen.generate(seed=43)

        # At least one thing should be different
        assert (
            not np.allclose(result1["object_positions"], result2["object_positions"])
            or result1["physics_params"] != result2["physics_params"]
        )
