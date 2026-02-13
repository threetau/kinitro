"""Procedural generation utilities for anti-overfitting."""

from typing import Any

import numpy as np

from kinitro.types import ProceduralTaskResult


def randomize_positions(
    base: np.ndarray | list[float],
    rng: np.random.Generator,
    range_xyz: list[float] | np.ndarray,
) -> np.ndarray:
    """
    Randomize 3D positions within a range around base position.

    Args:
        base: Base position [x, y, z]
        rng: Numpy random generator
        range_xyz: Range for each axis [dx, dy, dz]

    Returns:
        Randomized position as numpy array
    """
    base = np.asarray(base, dtype=np.float32)
    range_xyz = np.asarray(range_xyz, dtype=np.float32)

    offset = rng.uniform(-range_xyz, range_xyz)
    return base + offset


def randomize_physics(
    rng: np.random.Generator,
    param_ranges: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """
    Generate randomized physics parameters.

    Args:
        rng: Numpy random generator
        param_ranges: Dict of {param_name: (min, max)} ranges

    Returns:
        Dict of randomized physics parameters
    """
    return {name: float(rng.uniform(low, high)) for name, (low, high) in param_ranges.items()}


def randomize_domain(
    rng: np.random.Generator,
    config: dict[str, dict[str, Any]],  # Any: randomization specs vary by param type
) -> dict[str, Any]:  # Any: generated values are floats, ints, or tuples
    """
    Generate domain randomization parameters.

    Args:
        rng: Numpy random generator
        config: Dict of {param_name: {"type": ..., ...}} specifications

    Returns:
        Dict of randomized domain parameters
    """
    result = {}

    for name, spec in config.items():
        param_type = spec.get("type", "uniform")

        if param_type == "uniform":
            low, high = spec["range"]
            result[name] = float(rng.uniform(low, high))

        elif param_type == "choice":
            options = spec["options"]
            result[name] = rng.choice(options)

        elif param_type == "normal":
            mean, std = spec["mean"], spec["std"]
            result[name] = float(rng.normal(mean, std))

        elif param_type == "boolean":
            prob = spec.get("prob", 0.5)
            result[name] = bool(rng.random() < prob)

    return result


class ProceduralTaskGenerator:
    """
    Generates procedural task variations for an environment.

    This is the core anti-overfitting mechanism. Each evaluation uses
    freshly generated task instances that miners cannot have seen before.
    """

    def __init__(
        self,
        env_id: str,
        position_ranges: dict[str, np.ndarray] | None = None,
        physics_ranges: dict[str, tuple[float, float]] | None = None,
        domain_config: dict[str, dict[str, Any]] | None = None,  # Any: see randomize_domain
    ):
        """
        Initialize generator with randomization parameters.

        Args:
            env_id: Environment identifier
            position_ranges: {object_name: [dx, dy, dz]} position ranges
            physics_ranges: {param_name: (min, max)} physics ranges
            domain_config: Domain randomization configuration
        """
        self.env_id = env_id
        self.position_ranges = position_ranges or {}
        self.physics_ranges = physics_ranges or {
            "friction": (0.5, 1.5),
            "damping": (0.8, 1.2),
            "mass_scale": (0.8, 1.2),
        }
        self.domain_config = domain_config or {
            "camera_fov": {"type": "uniform", "range": (40, 60)},
            "light_intensity": {"type": "uniform", "range": (0.8, 1.2)},
        }

    def generate(
        self,
        seed: int,
        base_object_pos: np.ndarray | None = None,
        base_target_pos: np.ndarray | None = None,
    ) -> ProceduralTaskResult:
        """
        Generate procedural task parameters.

        Args:
            seed: Random seed
            base_object_pos: Base object position to randomize around
            base_target_pos: Base target position to randomize around

        Returns:
            Dict with object_positions, target_positions, physics_params, domain_randomization
        """
        rng = np.random.default_rng(seed)

        # Randomize object positions
        object_positions = base_object_pos if base_object_pos is not None else np.zeros(3)
        if "object" in self.position_ranges:
            object_positions = randomize_positions(
                object_positions, rng, self.position_ranges["object"]
            )

        # Randomize target positions
        target_positions = base_target_pos if base_target_pos is not None else np.zeros(3)
        if "target" in self.position_ranges:
            target_positions = randomize_positions(
                target_positions, rng, self.position_ranges["target"]
            )

        # Randomize physics
        physics_params = randomize_physics(rng, self.physics_ranges)

        # Domain randomization
        domain_randomization = randomize_domain(rng, self.domain_config)

        return {
            "object_positions": object_positions,
            "target_positions": target_positions,
            "physics_params": physics_params,
            "domain_randomization": domain_randomization,
        }
