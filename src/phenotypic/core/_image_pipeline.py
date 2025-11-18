from ._pipeline_parts._serializable_pipeline import SerializablePipeline


class ImagePipeline(SerializablePipeline):
    """
    A comprehensive class for sequential operation of image processing operations and measurements on images.

    This provides a high-level interface for applying image processing operations and
    extracting measurements from single images or image sets. Operations are applied sequentially to each image,
    followed by measurement extraction.

    The pipeline supports benchmarking and verbose logging to track execution performance and progress.

    Attributes:
        benchmark (bool): Whether to enable execution time tracking for operations and measurements.
        verbose (bool): Whether to enable verbose logging during pipeline execution.

    ** Example **

    .. code-block:: python
       >>> import phenotypic as pt
       >>> from phenotypic.detect import OtsuDetector
       >>> from phenotypic.measure import MeasureShape, MeasureIntensity
       >>>
       >>> pipe = pt.ImagePipeline(ops=[OtsuDetector()], meas=[MeasureShape(), MeasureIntensity()])

    """
    pass
