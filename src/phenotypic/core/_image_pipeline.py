from ._pipeline_parts._image_pipeline_core import ImagePipelineCore


class ImagePipeline(ImagePipelineCore):
    """
    A comprehensive class for sequential operation of image processing operations and measurements on images.

    This class inherits from `ImagePipelineCore` and provides a high-level interface for applying
    image processing operations and extracting measurements from single images or image sets.
    Operations are applied sequentially to each image, followed by measurement extraction.

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
       >>>
       >>> # Single image
       >>> img = pt.Image(pt.data.load_colony(), name="Colony")
       >>> meas = pipe.apply_and_measure(img)
       >>> meas.head()
       >>>
       >>> # For batch processing with ImageSet, use ImagePipelineBatch instead
       >>> # image_set = pt.ImageSet(name='example', grid_finder=pt.grid.AutoGridFinder(),
       >>> #                        src=[pt.data.load_plate_12hr(), pt.data.load_plate_72hr()])
       >>> # batch_pipe = pt.ImagePipelineBatch(ops=[OtsuDetector()], meas=[MeasureShape(), MeasureIntensity()])
       >>> # meas = batch_pipe.apply_and_measure(image_set)

    """
    pass
