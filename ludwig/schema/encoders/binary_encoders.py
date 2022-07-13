from typing import ClassVar

from marshmallow_dataclass import dataclass

from ludwig.encoders.base import Encoder
from ludwig.encoders.binary_encoders import BinaryPassthroughEncoder
from ludwig.schema import utils as schema_utils


@dataclass
class BinaryPassthroughEncoderConfig(schema_utils.BaseMarshmallowConfig):

    encoder_class: ClassVar[Encoder] = BinaryPassthroughEncoder

    type: str = "passthrough"
