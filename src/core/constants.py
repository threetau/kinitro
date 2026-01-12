from datetime import timedelta
from enum import StrEnum


class NeuronType(StrEnum):
    Miner = "miner"
    Validator = "validator"


class ImageFormat(StrEnum):
    PNG = "png"
    JPEG = "jpeg"


PRESIGN_EXPIRY = timedelta(days=7)
