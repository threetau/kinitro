import enum
from datetime import datetime
from typing import Any

from pydantic import GetCoreSchemaHandler, ValidationError
from pydantic_core import CoreSchema, core_schema
from sqlalchemy import text
from sqlalchemy.sql import func
from sqlmodel import Field as SQLModelField
from sqlmodel import SQLModel


class SnowflakeId:
    """
    Immutable newtype for Snowflake IDs.

    Stores 64-bit integer values internally, but serializes to strings in
    JSON/API contexts to prevent JavaScript precision loss (JS safe integer
    range is +/- 2^53-1).

    Supports comparison operations, hashing, and automatic Pydantic validation.
    """

    __slots__ = ("_value",)

    def __init__(self, value: int | str):
        """
        Initialize a SnowflakeId.

        Args:
            value: Integer or string representation of a snowflake ID (0 to 2^63-1)

        Raises:
            ValueError: If value is out of range or invalid
        """
        if isinstance(value, str):
            try:
                int_value = int(value)
            except ValueError:
                raise ValueError(f"Invalid SnowflakeId string: {value!r}")
        elif isinstance(value, int):
            int_value = value
        else:
            raise TypeError(
                f"SnowflakeId must be int or str, not {type(value).__name__}"
            )

        if not (0 <= int_value <= (2**63 - 1)):
            raise ValueError(
                f"SnowflakeId must be between 0 and 2^63-1, got {int_value}"
            )

        object.__setattr__(self, "_value", int_value)

    def __int__(self) -> int:
        """Return the underlying integer value."""
        return self._value

    def __str__(self) -> str:
        """Return string representation of the ID (numeric string for JSON)."""
        return str(self._value)

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return f"SnowflakeId({self._value})"

    def __eq__(self, other: object) -> bool:
        """Check equality with another SnowflakeId or int."""
        if isinstance(other, SnowflakeId):
            return self._value == other._value
        elif isinstance(other, int):
            return self._value == other
        return NotImplemented

    def __lt__(self, other: "SnowflakeId | int") -> bool:
        """Less than comparison."""
        if isinstance(other, SnowflakeId):
            return self._value < other._value
        elif isinstance(other, int):
            return self._value < other
        return NotImplemented

    def __le__(self, other: "SnowflakeId | int") -> bool:
        """Less than or equal comparison."""
        if isinstance(other, SnowflakeId):
            return self._value <= other._value
        elif isinstance(other, int):
            return self._value <= other
        return NotImplemented

    def __gt__(self, other: "SnowflakeId | int") -> bool:
        """Greater than comparison."""
        if isinstance(other, SnowflakeId):
            return self._value > other._value
        elif isinstance(other, int):
            return self._value > other
        return NotImplemented

    def __ge__(self, other: "SnowflakeId | int") -> bool:
        """Greater than or equal comparison."""
        if isinstance(other, SnowflakeId):
            return self._value >= other._value
        elif isinstance(other, int):
            return self._value >= other
        return NotImplemented

    def __hash__(self) -> int:
        """Return hash of the underlying value for use in sets/dicts."""
        return hash(self._value)

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent modification (immutability)."""
        raise AttributeError("SnowflakeId is immutable")

    def __delattr__(self, name: str) -> None:
        """Prevent deletion (immutability)."""
        raise AttributeError("SnowflakeId is immutable")

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """
        Pydantic v2 schema for validation and serialization.

        - Accepts int or str inputs
        - Validates range (0 to 2^63-1)
        - Serializes to string in JSON mode
        - Uses BigInteger for SQLAlchemy columns
        """

        def validate(value: int | str) -> "SnowflakeId":
            """Validate and construct SnowflakeId."""
            if isinstance(value, SnowflakeId):
                return value
            try:
                return cls(value)
            except (ValueError, TypeError) as e:
                raise ValidationError(str(e))

        def serialize(instance: "SnowflakeId", info) -> str | int:
            """Serialize to string for JSON, int for other contexts."""
            # In JSON mode, always use string to prevent precision loss
            if info.mode == "json":
                return str(instance._value)
            # For Python mode (e.g., model_dump()), also use string for consistency
            return str(instance._value)

        python_schema = core_schema.no_info_plain_validator_function(validate)

        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_plain_validator_function(validate),
            python_schema=python_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize, info_arg=True, return_schema=core_schema.str_schema()
            ),
        )

    @classmethod
    def __get_validators__(cls):
        """Pydantic v1 compatibility (if needed)."""
        yield cls.validate

    @classmethod
    def validate(cls, v):
        """Pydantic v1 validator."""
        return cls(v)


class EvaluationStatus(enum.Enum):
    """Evaluation job status enum."""

    QUEUED = "QUEUED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"


class TimestampMixin(SQLModel):
    """Mixin for created/updated timestamps."""

    created_at: datetime = SQLModelField(
        default=None,
        nullable=False,
        sa_column_kwargs={"server_default": text("CURRENT_TIMESTAMP"), "index": True},
    )
    updated_at: datetime = SQLModelField(
        default=None,
        nullable=False,
        sa_column_kwargs={
            "server_default": text("CURRENT_TIMESTAMP"),
            "onupdate": func.now(),
        },
    )
