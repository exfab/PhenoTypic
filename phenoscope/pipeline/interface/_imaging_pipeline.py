import logging

formatter = logging.Formatter(
        fmt=f'[%(asctime)s|%(name)s] %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S'
)
console_handler = logging.StreamHandler()
log = logging.getLogger(__name__)
log.addHandler(console_handler)
console_handler.setFormatter(formatter)

from typing import Dict, Any

from ...interface._image_operation import ImageOperation

from ...interface import ObjectDetector
from ...interface import ImagePreprocessor
from ...interface import MorphologyMorpher
from ...interface import ImageTransformer
from ...interface import MapModifier
from ...interface import FeatureExtractor
from ...grid.interface import GridFeatureExtractor
from ...interface._feature_extractor import FeatureExtractor

from ...util.constants import INTERFACE_ERROR_MSG


class ImagingPipeline(ImageOperation):
    def __init__(self, pipeline_queue: Dict[str, ImageOperation]):
        self._operational_queue: Dict[str, ImageOperation] = pipeline_queue

    def _execute_pipeline(self, input: Any, inplace: bool) -> Any:
        output = input
        for key in self._operational_queue.keys():
            log.info(f'Executing image operation named: {key}\n')
            output = self._execute_operation(output, self._operational_queue[key], inplace=inplace)

        return output

    def _execute_operation(self, input: Any, operator: ImageOperation, inplace: bool) -> Any:
        match operator:
            case ImagePreprocessor():
                return operator.preprocess(input, inplace=inplace)
            case MorphologyMorpher():
                return operator.morph(input, inplace=inplace)
            case ObjectDetector():
                return operator.detect(input, inplace=inplace)
            case ImageTransformer():
                return operator.transform(input, inplace=inplace)
            case MapModifier():
                return operator.modify(input, inplace=inplace)
            case FeatureExtractor():
                raise ValueError(
                        'Measurer objects cannot be a part of an imaging process pipeline. Place them in a measurer pipeline object instead.')
            case _:
                raise TypeError(f'Operator {type(operator)} not recognized')
