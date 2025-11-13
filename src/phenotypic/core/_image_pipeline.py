from ._pipeline_parts._image_pipeline_batch import ImagePipelineBatch


class ImagePipeline(ImagePipelineBatch):
    """
    A comprehensive class that allows for sequential operation of image processing operations and measurements on images.
    This can be run on a single `phenotypic.Image` at a time, or multiple images simultaneously using `phenotypic.ImageSet`.
    When using `phenotypic.ImageSet`, the pipeline can process all images in parallel natively, depending on the number of workers and the nature of the tasks involved.

    Attributes:
        num_workers (int): Number of worker processes for parallel execution. Must be at least 2 for parallel execution
        verbose (bool): Whether to enable verbose logging.
        memblock_factor (float): Adjustment factor for memory allocation during operations.
        timeout (Optional[int]): Time limit for joining threads during multi-threaded execution.

    ** Example **

    .. code-block:: python
       >>> import phenotypic as pt
       >>> from phenotypic.detect import OtsuDetector()
       >>> from phenotypic.measure import MeasureShape, MeasureIntensity()
       >>>
       >>> pipe = pt.ImagePipeline(ops=[OtsuDetector()], meas=[MeasureShape(), MeasureIntensity()])
       >>>
       >>> # Single image
       >>> img = pt.Image(pt.data.load_colony(), name="Colony")
       >>> meas = pipe.apply_and_measure(img)
       >>> meas.head()
       >>>
       >>> image_set = pt.ImageSet(
       >>>      name='example', grid_finder=pt.grid.AutoGridFinder(),
       >>>      src=[pt.data.load_plate_12hr(), pt.data.load_plate_72hr()]
       >>> )
       >>> pipe_meas = pipe.apply_and_measure(image_set)
       >>> meas.head()


    """
    pass
