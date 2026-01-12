"""Time and duration utilities for type-safe duration handling.

This module provides utilities for working with timedelta objects throughout the codebase,
including Pydantic v2 validators and serializers for converting between seconds (stored in
config/DB) and timedelta objects (used in code).
"""

from datetime import timedelta
from typing import Any, Union

from pydantic import field_serializer, field_validator
from pydantic_core.core_schema import FieldSerializationInfo


def seconds_to_timedelta(seconds: Union[int, float]) -> timedelta:
    """Convert seconds (int or float) to a timedelta object.

    Args:
        seconds: Number of seconds as int or float

    Returns:
        timedelta object representing the duration
    """
    return timedelta(seconds=seconds)


def timedelta_to_seconds(duration: timedelta) -> float:
    """Convert a timedelta object to seconds (float).

    Args:
        duration: timedelta object

    Returns:
        Total seconds as float
    """
    return duration.total_seconds()


class DurationField:
    """Mixin class providing Pydantic v2 validators and serializers for duration fields.

    Use this as a base class or copy the decorators to fields that should be stored as
    seconds in config/DB but represented as timedelta in code.

    Example:
        class MyConfig(BaseModel):
            timeout: timedelta

            @field_validator('timeout', mode='before')
            @classmethod
            def validate_timeout(cls, v: Any) -> timedelta:
                if isinstance(v, timedelta):
                    return v
                if isinstance(v, (int, float)):
                    return timedelta(seconds=v)
                raise ValueError(f"Invalid timeout value: {v}")

            @field_serializer('timeout')
            def serialize_timeout(self, value: timedelta, _info: FieldSerializationInfo) -> float:
                return value.total_seconds()
    """

    @staticmethod
    def validator(field_name: str):
        """Create a field validator for duration fields.

        Args:
            field_name: Name of the field to validate

        Returns:
            Pydantic field_validator decorator
        """

        @field_validator(field_name, mode="before")
        @classmethod
        def _validate_duration(cls, v: Any) -> timedelta:
            if isinstance(v, timedelta):
                return v
            if isinstance(v, (int, float)):
                return timedelta(seconds=v)
            raise ValueError(f"Invalid duration value for {field_name}: {v}")

        return _validate_duration

    @staticmethod
    def serializer(field_name: str):
        """Create a field serializer for duration fields.

        Args:
            field_name: Name of the field to serialize

        Returns:
            Pydantic field_serializer decorator
        """

        @field_serializer(field_name)
        def _serialize_duration(
            value: timedelta, _info: FieldSerializationInfo
        ) -> float:
            return value.total_seconds()

        return _serialize_duration
