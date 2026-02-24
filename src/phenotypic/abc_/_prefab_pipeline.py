from .._core._image_pipeline import ImagePipeline


class PrefabPipeline(ImagePipeline):
    """Marker class for pre-built, validated image processing pipelines from the PhenoTypic team.

    PrefabPipeline is a specialized subclass of ImagePipeline that distinguishes "official" pre-built
    pipelines maintained by the PhenoTypic development team from user-created custom pipelines. It
    serves as a marker class (no additional functionality) that signals "this pipeline is validated,
    documented, and recommended for specific use cases in microbe colony phenotyping."

    **What is PrefabPipeline?**

    PrefabPipeline is NOT an operation ABC and does NOT inherit from BaseOperation. Instead, it's a
    subclass of ImagePipeline that:

    - **Is a marker class:** Inherits all ImagePipeline functionality unchanged; no new methods.
    - **Indicates official status:** Subclasses of PrefabPipeline are pre-built, validated pipelines
      with documented performance, parameter settings, and recommended use cases.
    - **Enables classification:** Code can distinguish official pipelines (``isinstance(obj, PrefabPipeline)``)
      from user-defined pipelines for documentation, discovery, or defaulting.
    - **Provides templates:** Each PrefabPipeline subclass is a complete processing workflow (enhancement,
      detection, refinement, measurement) ready to use out-of-the-box.

    **Quick Decision Guide: PrefabPipeline vs Custom ImagePipeline**

    - Use PrefabPipeline for standard colony phenotyping on agar plates with validated workflows
    - Use custom ImagePipeline for novel imaging scenarios, optimization experiments, or specialized workflows
    - Start with PrefabPipeline to understand the pipeline structure, then customize via parameter tuning
    - Clone and modify a PrefabPipeline subclass when you need significant algorithm changes
    - Combine multiple PrefabPipelines sequentially for complex multi-stage workflows

    **Available PrefabPipeline Subclasses**

    The PhenoTypic team maintains several pre-built pipelines optimized for different imaging scenarios:

    1. **[HeavyOtsuPipeline](src/phenotypic/prefab/_heavy_otsu_pipeline.py):** Multi-layer Otsu detection with aggressive refinement.
       Robust detection on challenging images (uneven lighting, varied sizes). Computationally expensive.

    2. **[HeavyWatershedPipeline](src/phenotypic/prefab/_heavy_watershed_pipeline.py):** Watershed segmentation for separated colonies.
       Handles closely-spaced or merged colonies. Very expensive; for small batches or deep analysis.

    3. **[RoundPeaksPipeline](src/phenotypic/prefab/_round_peaks_pipeline.py):** Peak detection for circular, well-separated colonies.
       Fast and suitable for high-throughput screening of early-time-point growth.

    4. **[GridSectionPipeline](src/phenotypic/prefab/_grid_section_pipeline.py):** Per-well section extraction and fine-grained analysis.
       Moderate cost; enables per-well quality control and segmentation.

    **When to use PrefabPipeline vs Custom ImagePipeline**

    - **Use PrefabPipeline if:**
      - You're analyzing colony growth on agar plates (the intended use case).
      - You want an immediately usable, tested workflow without configuration.
      - You want to reproduce results matching published benchmarks or team documentation.
      - You need a baseline for custom extensions (subclass or copy and modify).

    - **Create a custom ImagePipeline if:**
      - Your imaging scenario is novel (unusual plate format, different organisms, special preparation).
      - You want to experiment with different detector/refiner/measurement combinations.
      - You have labeled ground truth and want to optimize parameters for your specific images.
      - You need pipeline extensions (custom operations not in standard library).

    **Using a PrefabPipeline**

    PrefabPipeline subclasses are used exactly like ImagePipeline:

    .. code-block:: python

        from phenotypic import Image, GridImage
        from phenotypic.prefab import HeavyOtsuPipeline

        # Load image(s)
        image = GridImage.imread('plate.jpg', nrows=8, ncols=12)

        # Instantiate and apply pipeline
        pipeline = HeavyOtsuPipeline()
        result = pipeline.apply(image)  # or .operate([image])

        # Access results
        colonies = result.objects
        measurements = result.measurements
        print(f"Detected: {len(colonies)} colonies")
        print(f"Measurements shape: {measurements.shape}")

    **Customizing a PrefabPipeline**

    PrefabPipelines accept tunable parameters in ``__init__()`` to adapt to your images without
    rebuilding the pipeline structure:

    .. code-block:: python

        from phenotypic.prefab import HeavyOtsuPipeline

        # Use defaults (recommended for most cases)
        pipeline1 = HeavyOtsuPipeline()

        # Tune for noisier images
        pipeline2 = HeavyOtsuPipeline(
            gaussian_sigma=7,                    # Stronger blur
            small_object_min_size=150,           # More aggressive noise removal
            border_remover_size=2                # Remove more edge objects
        )

        # Parameters are typically named after the algorithm or parameter they control.
        # See pipeline docstring for available parameters and typical values.

    **When Parameters Fail: Creating a Custom Pipeline**

    If PrefabPipeline parameter tuning doesn't solve your problem:

    1. **Analyze failures:** Which step fails (detection, refinement, measurement)?
      - Use ``pipeline.benchmark=True, verbose=True`` to trace execution.
      - Visually inspect intermediate results (detection masks, refined masks).

    2. **Create a custom pipeline:**

    .. code-block:: python

        from phenotypic import ImagePipeline
        from phenotypic.enhance import GaussianBlur, CLAHE
        from phenotypic.detect import CannyDetector  # Different detector
        from phenotypic.refine import SmallObjectRemover, MaskFill
        from phenotypic.measure import MeasureShape, MeasureColor

        # Custom pipeline for your specific use case
        custom = ImagePipeline()
        custom.add(GaussianBlur(sigma=3))
        custom.add(CLAHE())
        custom.add(CannyDetector(sigma=1.5, low_threshold=0.1, high_threshold=0.4))
        custom.add(SmallObjectRemover(min_size=100))
        custom.add(MaskFill())
        custom.add(MeasureShape())
        custom.add(MeasureColor())

        # Test and iterate
        result = custom.operate([image])

    3. **Share successful custom pipelines:** If you develop a successful custom pipeline for a new
       imaging scenario, consider contributing it as a PrefabPipeline subclass to the project.

    **Pipeline Composition Pattern**

    Combine multiple PrefabPipelinesor mix PrefabPipeline with custom operations:

    .. code-block:: python

        from phenotypic import ImagePipeline
        from phenotypic.prefab import HeavyOtsuPipeline, RoundPeaksPipeline
        from phenotypic.refine import SmallObjectRemover
        from phenotypic.measure import MeasureIntensity

        # Combine different detection strategies with shared refinement
        pipeline = ImagePipeline()
        pipeline.add(HeavyOtsuPipeline())  # First detection attempt
        pipeline.add(SmallObjectRemover(min_size=100))  # Noise removal
        pipeline.add(MeasureIntensity())  # Measurement

        # Apply to image
        result = pipeline.apply(image)

    **Pipeline Serialization Pattern**

    Save and load pipelines for reproducible batch processing:

    .. code-block:: python

        from phenotypic.prefab import HeavyOtsuPipeline

        # Create, configure, and save
        pipeline = HeavyOtsuPipeline(gaussian_sigma=2.0, small_object_min_size=150)
        pipeline.to_json('my_colony_pipeline.json')  # Save configuration
        # pipeline.to_yaml('my_colony_pipeline.yaml')  # Alternative format

        # Load for batch processing (reproducible results)
        loaded = HeavyOtsuPipeline.from_json('my_colony_pipeline.json')
        results = loaded.operate([image1, image2, image3])

    **Extending PrefabPipeline**

    To create a new official PrefabPipeline subclass:

    .. code-block:: python

        from phenotypic.abc_ import PrefabPipeline
        from phenotypic.enhance import GaussianBlur, CLAHE
        from phenotypic.detect import OtsuDetector
        from phenotypic.refine import SmallObjectRemover
        from phenotypic.measure import MeasureShape

        class MyCustomPrefabPipeline(PrefabPipeline):
            '''Brief description of when to use this pipeline.'''

            def __init__(self, param1: int = 100, param2: float = 1.5,
                         benchmark: bool = False, verbose: bool = False):
                '''Initialize with tunable parameters.'''
                pipe_cfgs = [
                    GaussianBlur(sigma=param2),
                    CLAHE(),
                    OtsuDetector(),
                    SmallObjectRemover(min_size=param1),
                ]
                meas = [MeasureShape()]
                super().__init__(pipe_cfgs=pipe_cfgs, meas=meas, benchmark=benchmark,
                               verbose=verbose)

    Notes:
        - **Is a marker, not an operation:** PrefabPipeline does not inherit from BaseOperation.
          It's a convenient subclass of ImagePipeline for classification and discovery.

        - **Inheritance of ImagePipeline features:** PrefabPipeline inherits all ImagePipeline
          functionality: sequential operation chaining, benchmarking, verbose logging, batch
          processing via ``.operate()``, and serialization via ``.to_yaml()`` / ``.from_yaml()``.

        - **Parameter tuning via __init__():** Most PrefabPipeline subclasses expose key algorithm
          parameters in ``__init__()`` (e.g., detection threshold, smoothing sigma, refinement
          shape). Adjust these for your specific images before scaling to large batches.

        - **Benchmarking for profiling:** Set ``benchmark=True`` when instantiating to track
          execution time and memory usage per operation. Useful for identifying bottlenecks in
          large batch runs.

        - **Documentation and examples:** Each PrefabPipeline subclass is documented with use cases,
          typical parameters, performance characteristics, and example code. Check the subclass
          docstring for guidance.

        - **Not for operations:** Use PrefabPipeline only for complete pipelines. For individual
          operations (detection, enhancement, measurement), use operation ABCs directly.

    Examples:
        Quick start: Detect colonies with HeavyOtsuPipeline:

        >>> from phenotypic import GridImage
        >>> from phenotypic.prefab import HeavyOtsuPipeline
        >>> # Load a 96-well plate image
        >>> image = GridImage.imread('agar_plate.jpg', nrows=8, ncols=12)
        >>> # Use the pre-built, validated pipeline
        >>> pipeline = HeavyOtsuPipeline()
        >>> result = pipeline.apply(image)
        >>> # Access results
        >>> print(f"Detected {len(result.objects)} colonies")
        >>> print(f"Measurements: {result.measurements.columns.tolist()}")

        Batch processing multiple plates with a PrefabPipeline:

        >>> from phenotypic import GridImage
        >>> from phenotypic.prefab import HeavyOtsuPipeline
        >>> import glob
        >>> # Load multiple plate images
        >>> image_paths = glob.glob('batch_*.jpg')
        >>> images = [GridImage.imread(p, nrows=8, ncols=12)
        ...           for p in image_paths]
        >>> # Create pipeline (reusable for all images)
        >>> pipeline = HeavyOtsuPipeline(benchmark=True)
        >>> # Batch process
        >>> results = pipeline.operate(images)
        >>> # Collect results
        >>> for i, result in enumerate(results):
        ...     print(f"Image {i}: {len(result.objects)} colonies")
        ...     print(f"Measurements shape: {result.measurements.shape}")

        Customizing pipeline parameters for difficult images:

        >>> from phenotypic import GridImage
        >>> from phenotypic.prefab import HeavyOtsuPipeline
        >>> image = GridImage.imread('noisy_plate.jpg', nrows=8, ncols=12)
        >>> # Increase smoothing and noise removal for difficult images
        >>> pipeline = HeavyOtsuPipeline(
        ...     gaussian_sigma=8,                      # Stronger blur
        ...     small_object_min_size=200,             # Aggressive noise removal
        ...     border_remover_size=2                  # More border filtering
        ... )
        >>> result = pipeline.apply(image)
        >>> print(f"Robust detection: {len(result.objects)} colonies")

        Comparing PrefabPipeline vs custom pipeline:

        >>> from phenotypic import GridImage, ImagePipeline
        >>> from phenotypic.prefab import HeavyOtsuPipeline
        >>> from phenotypic.detect import CannyDetector
        >>> from phenotypic.refine import SmallObjectRemover
        >>> image = GridImage.imread('plate.jpg', nrows=8, ncols=12)
        >>> # Option 1: Use pre-built validated pipeline
        >>> prefab = HeavyOtsuPipeline()
        >>> result1 = prefab.apply(image)
        >>> # Option 2: Create custom pipeline for comparison
        >>> custom = ImagePipeline()
        >>> from phenotypic.enhance import GaussianBlur
        >>> custom.add(GaussianBlur(sigma=2))
        >>> custom.add(CannyDetector(sigma=1.5, low_threshold=0.1, high_threshold=0.4))
        >>> custom.add(SmallObjectRemover(min_size=100))
        >>> result2 = custom.apply(image)
        >>> # Compare results
        >>> print(f"Prefab: {len(result1.objects)}, Custom: {len(result2.objects)}")
    """

    pass
