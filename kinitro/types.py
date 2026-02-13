"""
Cross-module type definitions for Kinitro.

This module centralizes NewTypes, type aliases, enums, and TypedDicts
used across multiple Kinitro modules. It has zero kinitro imports
to avoid circular dependency risk.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, NamedTuple, NewType, NotRequired, Protocol, TypeAlias, TypedDict

import numpy as np

MinerUID = NewType("MinerUID", int)
BlockNumber = NewType("BlockNumber", int)
EnvironmentId = NewType("EnvironmentId", str)
TaskUUID = NewType("TaskUUID", str)
Hotkey = NewType("Hotkey", str)
Seed = NewType("Seed", int)

MinerScores: TypeAlias = dict[MinerUID, dict[EnvironmentId, float]]  # uid -> env_id -> score
MinerThresholds: TypeAlias = dict[
    MinerUID, dict[EnvironmentId, float]
]  # uid -> env_id -> threshold
MinerFirstBlocks: TypeAlias = dict[MinerUID, BlockNumber]  # uid -> block_number


def env_family_from_id(env_id: str) -> str:
    """Extract the family prefix from an environment ID.

    Example: ``"metaworld/pick-place-v3"`` → ``"metaworld"``.
    """
    return env_id.split("/", maxsplit=1)[0] if "/" in env_id else env_id


class EnvironmentFamily(str, Enum):
    """Environment family discriminator."""

    METAWORLD = "metaworld"
    GENESIS = "genesis"


class ObjectType(Enum):
    """Primitive object types for Genesis scenes."""

    BOX = "box"
    SPHERE = "sphere"
    CYLINDER = "cylinder"


class SubsetWeightScheme(Enum):
    """Weight scheme for environment subsets."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EQUAL = "equal"


class EncodedImage(TypedDict):
    """Base64-encoded image with metadata, returned by encode_image()."""

    data: str
    shape: list[int]
    dtype: str


class ParsedCommitment(TypedDict):
    """Parsed commitment fields from parse_commitment()."""

    huggingface_repo: str
    revision_sha: str
    deployment_id: str
    encrypted_deployment: str | None
    docker_image: str


class TaskResultData(TypedDict):
    """Inline result dict stored in TaskPoolORM.result."""

    success: bool
    score: float
    total_reward: float
    timesteps: int
    error: str | None


class MinerScoreData(TypedDict):
    """Score data dict for add_miner_scores_bulk()."""

    uid: MinerUID
    hotkey: Hotkey
    env_id: EnvironmentId
    success_rate: float
    mean_reward: float
    episodes_completed: int
    episodes_failed: int


class TaskCreateData(TypedDict):
    """Task data dict for create_tasks_bulk()."""

    cycle_id: int
    miner_uid: MinerUID
    miner_hotkey: Hotkey
    miner_endpoint: str
    env_id: EnvironmentId
    seed: Seed
    task_uuid: NotRequired[TaskUUID]
    miner_repo: NotRequired[str | None]
    miner_revision: NotRequired[str | None]


class VerificationDetails(TypedDict):
    """Details dict for VerificationResult.details."""

    match_scores: list[float]
    test_seed: int
    num_samples: int


class StepInfo(TypedDict, total=False):
    """Info dict returned by step(). total=False because environments populate different subsets."""

    task_prompt: str
    task_type: str
    episode_steps: int
    success: bool
    fallen: bool


class FeasibilityResult(NamedTuple):
    """Result of check_task_feasibility(). Backwards-compatible with tuple unpacking."""

    feasible: bool
    reason: str


class EligibilityResult(NamedTuple):
    """Result of verify_weight_setting_eligibility(). Backwards-compatible with tuple unpacking."""

    eligible: bool
    reason: str


class EnvStatsEntry(TypedDict):
    """Per-environment statistics entry used in API routes."""

    count: int
    total_sr: float


class RobotStateDict(TypedDict):
    """Robot state dictionary used across Genesis base and environment classes."""

    base_pos: np.ndarray  # (3,)
    base_quat: np.ndarray  # (4,) wxyz
    base_vel: np.ndarray  # (3,)
    base_ang_vel: np.ndarray  # (3,)
    dof_pos: np.ndarray  # (n_dofs,)
    dof_vel: np.ndarray  # (n_dofs,)


class ProceduralTaskResult(TypedDict):
    """Result of ProceduralTaskGenerator.generate()."""

    object_positions: np.ndarray  # (n_objects, 3)
    target_positions: np.ndarray  # (n_objects, 3)
    physics_params: dict[str, float]
    # Any: env-specific randomization params vary by implementation
    domain_randomization: dict[str, Any]


class AffinetesEnv(Protocol):
    """Structural interface for affinetes-managed evaluation environments.

    Used by the executor subsystem to interact with Docker/Basilica
    environments without depending on the ``affinetes`` package at
    type-checking time.
    """

    def is_ready(self) -> bool: ...
    async def list_environments(self) -> list[EnvironmentId]: ...
    # Any kwargs/return: evaluation payloads are environment-specific and vary
    # by implementation (e.g. Genesis vs MetaWorld), so we cannot tighten these.
    async def evaluate(self, **kwargs: Any) -> dict[str, Any]: ...
    async def cleanup(self) -> None: ...
