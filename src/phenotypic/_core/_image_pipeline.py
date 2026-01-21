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

    Example:

    .. dropdown:: Create a pipeline with detector and measurements

        .. code-block:: python

            >>> import phenotypic as pht
            >>> from phenotypic.enhance import GaussianBlur
            >>> from phenotypic.detect import OtsuDetector
            >>> from phenotypic.measure import MeasureShape, MeasureIntensity
            >>>
            >>> pipe = pht.ImagePipeline(
            >>>     ops=[GaussianBlur(),OtsuDetector()],
            >>>     meas=[MeasureShape(), MeasureIntensity()]
            >>> )
            >>>
            >>> image = pht.data.load_synth_plate()
            >>> pipe.apply(image)

    """

    pass
