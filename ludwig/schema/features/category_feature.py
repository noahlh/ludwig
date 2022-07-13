from typing import Optional

from marshmallow_dataclass import dataclass

from ludwig.constants import CATEGORY
from ludwig.decoders.registry import get_decoder_classes
from ludwig.schema import utils as schema_utils
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.preprocessing import BasePreprocessingConfig, PreprocessingDataclassField
from ludwig.schema.encoders.encoders import BaseEncoderConfig, EncoderDataclassField


@dataclass
class CategoryInputFeatureConfig(BaseInputFeatureConfig):
    """CategoryInputFeature is a dataclass that configures the parameters used for a category input feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(
        feature_type=CATEGORY
    )

    encoder: BaseEncoderConfig = EncoderDataclassField(
        feature_type=CATEGORY,
        default='dense',
    )


@dataclass
class CategoryOutputFeatureConfig(BaseOutputFeatureConfig):
    """CategoryOutputFeature is a dataclass that configures the parameters used for a category output feature."""

    decoder: Optional[str] = schema_utils.StringOptions(
        list(get_decoder_classes(CATEGORY).keys()),
        default="classifier",
        allow_none=True,
        description="Decoder to use for this category feature.",
    )
