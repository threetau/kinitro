"""Lightweight protocol definitions for the Swarm PyBullet environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

from .constants import TaskType

try:  # Optional dependency; only needed when pack/unpack is used
    import msgpack  # type: ignore
except Exception:  # pragma: no cover - msgpack is optional
    msgpack = None


@dataclass(slots=True)
class MapTask:
    """Scenario description for the drone navigation task."""

    map_seed: int
    start: Tuple[float, float, float]
    goal: Tuple[float, float, float]
    sim_dt: float
    horizon: float
    challenge_type: TaskType
    payload_mass_factor: Optional[float] = None
    payload_com_offset: Optional[Tuple[float, float, float]] = None
    thrust_scale: Optional[float] = None
    drag_scale: Optional[float] = None
    wind_xy: Optional[Tuple[float, float]] = None
    action_latency: Optional[float] = None
    payload_enabled: Optional[bool] = None
    domain_randomization: Optional[bool] = None
    version: str = "1"

    def pack(self) -> bytes:
        """Serialize the task to bytes using msgpack if available."""
        if msgpack is None:
            msg = "msgpack is required to pack MapTask"
            raise ImportError(msg)
        return msgpack.packb(asdict(self), use_bin_type=True)

    @staticmethod
    def unpack(blob: bytes) -> "MapTask":
        """Deserialize a MapTask from msgpack bytes."""
        if msgpack is None:
            msg = "msgpack is required to unpack MapTask"
            raise ImportError(msg)
        return MapTask(**msgpack.unpackb(blob, raw=False))


__all__ = ["MapTask"]
