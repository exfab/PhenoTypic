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

from ...interface._object_detector import ObjectDetector
from ...interface._image_preprocessor import ImagePreprocessor
from ...interface._morphology_morpher import MorphologyMorpher
from ...interface._image_transformer import ImageTransformer
from ...interface._object_filter import ObjectFilter
from ...interface._feature_extractor import FeatureExtractor
from ...interface._object_modifier import ObjectModifier

from ...util.error_message import INTERFACE_ERROR_MSG


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
            case FeatureExtractor():
                return operator.measure(input, inplace=inplace)
            case ObjectFilter():
                return operator.filter(input, inplace=inplace)
            case ImageTransformer():
                return operator.transform(input, inplace=inplace)
            case ObjectModifier():
                return operator.modify(input, inplace=inplace)
            case FeatureExtractor():
                raise ValueError(
                        'Measurer objects cannot be a part of an imaging process pipeline. Place them in a measurer pipeline object instead.')
            case _:
                raise TypeError(f'Operator {type(operator)} not recognized')
