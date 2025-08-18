from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field

CHAIN_COMMITMENT_VERSION = "1.0"


class ModelProvider(StrEnum):
    HuggingFace = "HuggingFace"
    R2 = "R2"


class ChainCommitment(BaseModel):
    """
    Represents a chain commitment with a unique identifier and the commitment data.
    """

    version: str = Field(
        default=CHAIN_COMMITMENT_VERSION,
        description="Version of the chain commitment schema",
    )


class ModelChainCommitment(ChainCommitment):
    """
    Represents a model chain commitment with a unique identifier and the commitment data.
    """

    provider: ModelProvider = Field(..., description="Provider of the model")
    repo_id: str = Field(
        ..., description="Identifier for the repository on the provider"
    )


@dataclass
class ChainCommitmentResponse:
    hotkey: str
    data: ChainCommitment
