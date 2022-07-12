from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.schema import utils as schema_utils
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField
from ludwig.schema.encoders.encoders import BaseEncoderConfig, EncoderDataclassField


@dataclass
class TimeseriesInputFeatureConfig(schema_utils.BaseMarshmallowConfig):
    """
    TimeseriesInputFeatureConfig is a dataclass that configures the parameters used for a timeseries input feature.
    """

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(
        feature_type='timeseries'
    )

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type='timeseries',
        default='parallel_cnn',
    )

    # TODO(#1673): Need some more logic here for validating against input features
    tied: Optional[str] = schema_utils.String(
        default=None,
        allow_none=True,
        description="Name of input feature to tie the weights of the encoder with.  It needs to be the name of a "
                    "feature of the same type and with the same encoder parameters.",
    )
