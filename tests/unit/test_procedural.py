"""Tests for procedural generation."""

import uuid

import numpy as np

from kinitro.environments.procedural import (
    ProceduralTaskGenerator,
    randomize_physics,
    randomize_positions,
)
from kinitro.scheduler.task_generator import generate_seed


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


class TestGenerateSeed:
    """Tests for seed generation from task UUID."""

    def test_deterministic(self):
        """Same UUID should give same seed."""
        task_uuid = str(uuid.uuid4())
        seed1 = generate_seed(task_uuid)
        seed2 = generate_seed(task_uuid)

        assert seed1 == seed2

    def test_different_uuids_different_seeds(self):
        """Different UUIDs should give different seeds."""
        uuid1 = str(uuid.uuid4())
        uuid2 = str(uuid.uuid4())
        seed1 = generate_seed(uuid1)
        seed2 = generate_seed(uuid2)

        assert seed1 != seed2

    def test_positive_31bit(self):
        """Seed should be positive 31-bit integer (fits PostgreSQL int4)."""
        for _ in range(100):
            task_uuid = str(uuid.uuid4())
            seed = generate_seed(task_uuid)
            assert 0 <= seed <= 0x7FFFFFFF

    def test_consistent_across_calls(self):
        """Same UUID should produce same seed across multiple calls."""
        test_uuid = "550e8400-e29b-41d4-a716-446655440000"
        seeds = [generate_seed(test_uuid) for _ in range(10)]
        assert all(s == seeds[0] for s in seeds)


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
