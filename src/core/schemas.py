from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field


class ModelProvider(StrEnum):
    HF = "HF"  # Hugging Face
    R2 = "R2"  # Direct vault (Cloudflare R2 or S3-compatible)


class ChainCommitment(BaseModel):
    """
    Represents a model chain commitment with a unique identifier and the commitment data.
    """

    provider: ModelProvider = Field(..., description="Provider of the model")
    comp_id: str = Field(..., description="Identifier for the competition")
    repo_id: str = Field(
        ..., description="Identifier for the repository on the provider"
    )


@dataclass
class ChainCommitmentResponse:
    hotkey: str
    data: ChainCommitment
