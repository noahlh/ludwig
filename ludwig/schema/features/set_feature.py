from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.decoders.registry import get_decoder_classes

from ludwig.schema import utils as schema_utils
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField
from ludwig.schema.encoders.encoders import BaseEncoderConfig, EncoderDataclassField


@dataclass
class SetInputFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """
    SetInputFeatureConfig is a dataclass that configures the parameters used for a set input feature.
    """

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(
        feature_type='set'
    )

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type='set',
        default='embed',
    )

    # TODO(#1673): Need some more logic here for validating against input features
    tied: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
                    "feature of the same type and with the same encoder parameters.",
    )


@dataclass
class SetOutputFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """
    SetOutputFeatureConfig is a dataclass that configures the parameters used for a set output feature.
    """

    decoder: Optional[str] = schema_utils.StringOptions(
        list(get_decoder_classes('set').keys()),
        default="classifier",
        allow_none=True,
        description="Decoder to use for this set feature.",
    )
