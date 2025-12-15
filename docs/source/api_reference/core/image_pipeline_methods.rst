ImagePipeline Class Methods
===========================

.. toctree::
   :hidden:
   :maxdepth: 2
   :local:

   self

.. currentmodule:: phenotypic

Overview
--------

The :class:`ImagePipeline` class provides a comprehensive interface for sequential
execution of image processing operations and measurements on images. It manages
two queues: a processing queue containing operations applied sequentially to each
image, and a measurement queue containing feature extractors that analyze images
and produce results as pandas DataFrames.

The pipeline supports:

- **Sequential operations:** Apply detection, enhancement, refinement, and correction operations in order
- **Measurement extraction:** Extract shape, intensity, texture, and custom features
- **Performance monitoring:** Track execution time with benchmarking
- **Serialization:** Save/load pipeline configurations as JSON
- **Batch processing:** Process multiple images efficiently
- **Verbose logging:** Progress reporting during execution

Pipelines are the recommended approach for reproducible image analysis workflows,
ensuring consistent processing across image sets and facilitating parameter optimization.

Initialization & Configuration
-------------------------------

.. automethod:: ImagePipeline.__init__

Create a new pipeline with specified operations and measurements. Operations are
applied sequentially in the order provided. Measurements are extracted after all
operations complete.

**Parameters:**

- **ops:** List or dict of ImageOperation instances (detectors, enhancers, refiners)
- **meas:** List or dict of MeasureFeatures instances (shape, intensity, custom features)
- **benchmark:** Enable execution time tracking (default: False)
- **verbose:** Enable progress reporting and diagnostic output (default: False)

**Example:**

.. code-block:: python

    from phenotypic import ImagePipeline
    from phenotypic.detect import OtsuDetector
    from phenotypic.enhance import GaussianBlur
    from phenotypic.refine import RemoveSmallObjects
    from phenotypic.measure import MeasureShape, MeasureIntensity
    
    # Create pipeline with operations and measurements
    pipeline = ImagePipeline(
        ops=[
            GaussianBlur(sigma=1.5),
            OtsuDetector(),
            RemoveSmallObjects(min_size=100)
        ],
        meas=[
            MeasureShape(),
            MeasureIntensity()
        ],
        benchmark=True,
        verbose=True
    )
    
    # Apply to image
    img = Image.imread('plate.jpg')
    results = pipeline.apply_and_measure(img)

.. automethod:: ImagePipeline.set_ops

Register or update the operations queue. Operations can be provided as a list
(auto-named by class) or a dictionary (explicit names). If operations with duplicate
class names are provided, automatic suffix numbering ensures uniqueness.

**Example:**

.. code-block:: python

    pipeline = ImagePipeline()
    
    # Set operations as list
    pipeline.set_ops([
        GaussianBlur(sigma=1.5),
        OtsuDetector(),
        RemoveSmallObjects(min_size=100)
    ])
    # Names: 'GaussianBlur', 'OtsuDetector', 'RemoveSmallObjects'
    
    # Set operations as dict with custom names
    pipeline.set_ops({
        'blur': GaussianBlur(sigma=1.5),
        'detect': OtsuDetector(),
        'refine': RemoveSmallObjects(min_size=100)
    })

.. automethod:: ImagePipeline.set_meas

Register or update the measurements queue. Measurements can be provided as a list
(auto-named by class) or a dictionary (explicit names).

**Example:**

.. code-block:: python

    pipeline = ImagePipeline()
    
    # Set measurements as list
    pipeline.set_meas([
        MeasureShape(),
        MeasureIntensity(),
        MeasureTexture()
    ])
    
    # Set measurements as dict with custom names
    pipeline.set_meas({
        'shape': MeasureShape(),
        'intensity': MeasureIntensity(),
        'texture': MeasureTexture()
    })

Pipeline Execution
------------------

.. automethod:: ImagePipeline.apply

Apply all registered operations to an image sequentially. Operations are executed
in the order they were registered. Each operation receives the output of the previous
operation.

**Parameters:**

- **image:** Image instance to process
- **inplace:** If True, modify image directly; if False, create a copy (default: False)
- **reset:** If True, reset image state before applying pipeline (default: True)

**Returns:** Processed Image (or GridImage if input was GridImage)

**Example:**

.. code-block:: python

    # Apply pipeline to single image
    img = Image.imread('plate.jpg')
    processed = pipeline.apply(img, inplace=False)
    
    # Show before and after
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    img.show(ax=ax1)
    ax1.set_title('Original')
    processed.show_overlay(ax=ax2)
    ax2.set_title('Processed')

.. automethod:: ImagePipeline.measure

Extract measurements from a processed image using all registered measurement extractors.
Results from all measurements are merged into a single DataFrame indexed by object labels.

**Parameters:**

- **image:** Image instance to measure (must have detected objects)
- **include_metadata:** If True, include image metadata in results (default: True)

**Returns:** pandas DataFrame with measurements for each detected object

**Example:**

.. code-block:: python

    # Measure after detection
    detected = pipeline.apply(img)
    measurements = pipeline.measure(detected, include_metadata=True)
    
    # Inspect results
    print(measurements.head())
    print(measurements.columns)
    
    # Filter by measurement criteria
    large_colonies = measurements[measurements['area'] > 500]
    bright_colonies = measurements[measurements['mean_intensity'] > 150]

.. automethod:: ImagePipeline.apply_and_measure

Convenience method that applies all operations and then extracts all measurements
in a single call. Equivalent to calling ``apply()`` followed by ``measure()``.

**Parameters:**

- **image:** Image instance to process and measure
- **inplace:** If True, modify image directly (default: False)
- **reset:** If True, reset image state before processing (default: True)
- **include_metadata:** If True, include metadata in measurements (default: True)

**Returns:** pandas DataFrame with measurements for each detected object

**Example:**

.. code-block:: python

    # Process and measure in one call
    img = Image.imread('plate.jpg')
    results = pipeline.apply_and_measure(img)
    
    # Results ready for analysis
    import seaborn as sns
    sns.scatterplot(data=results, x='area', y='mean_intensity')

Performance Monitoring
----------------------

The ImagePipeline class provides comprehensive performance monitoring capabilities
through benchmarking and verbose logging. These features help identify bottlenecks
and optimize pipeline execution.

**Benchmark Flag:**

When ``benchmark=True``, the pipeline tracks execution time for each operation and
measurement. Times are stored internally and accessible via ``benchmark_results()``.

**Verbose Flag:**

When ``verbose=True`` and ``benchmark=True``, the pipeline prints progress information
during execution, including:

- Current operation/measurement being executed
- Execution time for each step
- Progress bar (if tqdm is installed)
- Total execution time

.. automethod:: ImagePipeline.benchmark_results

Return a DataFrame containing execution times for all operations and measurements
from the most recent pipeline run. This method should be called after executing
the pipeline with ``benchmark=True``.

**Returns:** pandas DataFrame with columns:

- **Process Type:** 'Operation' or 'Measurement'
- **Process Name:** Name of the operation/measurement
- **Execution Time (s):** Time in seconds

**Example:**

.. code-block:: python

    # Create pipeline with benchmarking enabled
    pipeline = ImagePipeline(
        ops=[GaussianBlur(), OtsuDetector(), RemoveSmallObjects()],
        meas=[MeasureShape(), MeasureIntensity()],
        benchmark=True,
        verbose=True
    )
    
    # Apply to image
    img = Image.imread('plate.jpg')
    results = pipeline.apply_and_measure(img)
    
    # Get performance metrics
    performance = pipeline.benchmark_results()
    print(performance)
    #   Process Type         Process Name  Execution Time (s)
    # 0    Operation         GaussianBlur            0.0234
    # 1    Operation        OtsuDetector            0.1523
    # 2    Operation  RemoveSmallObjects            0.0456
    # 3  Measurement         MeasureShape            0.0893
    # 4  Measurement     MeasureIntensity            0.0234
    # 5        Total        All Processes            0.3340
    
    # Identify bottlenecks
    slowest = performance.nlargest(3, 'Execution Time (s)')
    print(f"Slowest steps:\n{slowest}")

**Optimization Tips:**

Based on benchmark results:

- **Detection is slow:** Consider downsampling images or using faster detectors
- **Measurement is slow:** Reduce number of measured features or measure fewer objects
- **Multiple operations slow:** Consider parallelization with ImagePipelineBatch
- **File I/O slow:** Use HDF5 format instead of pickle for large image sets

**Progress Reporting:**

When both ``benchmark`` and ``verbose`` are enabled, progress is displayed during
execution:

.. code-block:: text

    Applying operations: 100%|████████████| 3/3 [00:00<00:00, 12.34it/s]
    Operation: OtsuDetector  time=0.1523s
    Measuring image properties...
    Applying measurements: 100%|████████| 2/2 [00:00<00:00, 16.78it/s]

Serialization & State
---------------------

The ImagePipeline class supports serialization to JSON format, enabling pipeline
configurations to be saved, shared, and reused across sessions. This ensures
reproducibility and facilitates collaboration.

.. automethod:: ImagePipeline.to_json

Serialize the pipeline configuration to JSON format. Captures operations, measurements,
and configuration flags (benchmark, verbose). Internal state and pandas DataFrames
are excluded to keep serialization clean.

**Parameters:**

- **filepath:** Optional path to save JSON (string or Path). If None, returns JSON string.

**Returns:** JSON string representation of pipeline configuration

**Serialization includes:**

- Operation instances with their parameters
- Measurement instances with their parameters
- Benchmark and verbose flags
- Operation/measurement order

**Serialization excludes:**

- Internal state (attributes starting with '_')
- pandas DataFrames (measurement results)
- Temporary computation results
- Cached data

**Example:**

.. code-block:: python

    # Create and configure pipeline
    pipeline = ImagePipeline(
        ops=[GaussianBlur(sigma=1.5), OtsuDetector()],
        meas=[MeasureShape(), MeasureIntensity()],
        benchmark=True,
        verbose=False
    )
    
    # Save to file
    pipeline.to_json('my_pipeline.json')
    
    # Get JSON string
    json_str = pipeline.to_json()
    print(json_str)

**Example JSON Output:**

.. code-block:: json

    {
      "ops": {
        "GaussianBlur": {
          "class": "GaussianBlur",
          "params": {"sigma": 1.5}
        },
        "OtsuDetector": {
          "class": "OtsuDetector",
          "params": {}
        }
      },
      "meas": {
        "MeasureShape": {
          "class": "MeasureShape",
          "params": {}
        },
        "MeasureIntensity": {
          "class": "MeasureIntensity",
          "params": {}
        }
      },
      "benchmark": true,
      "verbose": false
    }

.. automethod:: ImagePipeline.from_json

Deserialize a pipeline from JSON format. Reconstructs operations, measurements,
and configuration flags. Classes are imported from the phenotypic namespace and
instantiated with their saved parameters.

**Parameters:**

- **json_data:** Either a JSON string or a path to a JSON file

**Returns:** ImagePipeline instance with loaded configuration

**Raises:**

- **ValueError:** If JSON is invalid or cannot be parsed
- **ImportError:** If a required operation/measurement class cannot be imported
- **AttributeError:** If a class cannot be found in phenotypic namespace

**Example:**

.. code-block:: python

    # Load from file
    loaded_pipeline = ImagePipeline.from_json('my_pipeline.json')
    
    # Load from string
    json_str = '{"ops": {...}, "meas": {...}}'
    loaded_pipeline = ImagePipeline.from_json(json_str)
    
    # Use loaded pipeline
    img = Image.imread('plate.jpg')
    results = loaded_pipeline.apply_and_measure(img)

**Reproducibility Workflow:**

.. code-block:: python

    # Researcher 1: Develop and save pipeline
    pipeline = ImagePipeline(
        ops=[GaussianBlur(sigma=2.0), OtsuDetector()],
        meas=[MeasureShape()]
    )
    pipeline.to_json('colony_detection_v1.json')
    
    # Researcher 2: Load and apply to new data
    pipeline = ImagePipeline.from_json('colony_detection_v1.json')
    results = []
    for img_path in image_list:
        img = Image.imread(img_path)
        df = pipeline.apply_and_measure(img)
        results.append(df)
    
    # Guaranteed identical processing!

**Pipeline Versioning:**

.. code-block:: python

    # Save pipeline with version tracking
    import json
    from datetime import datetime
    
    config = json.loads(pipeline.to_json())
    config['version'] = '1.0'
    config['created'] = datetime.now().isoformat()
    config['description'] = 'Colony detection for strain X'
    
    with open('pipeline_v1.0.json', 'w') as f:
        json.dump(config, f, indent=2)

Configuration Persistence
~~~~~~~~~~~~~~~~~~~~~~~~~

Pipeline configurations can be version controlled alongside analysis code, ensuring
that:

- Processing parameters are documented and reproducible
- Changes to pipelines are tracked over time
- Collaborators use identical processing workflows
- Results can be regenerated with exact parameter settings

**Best Practices:**

1. **Version pipeline files:** Use descriptive names like ``detection_v1.0.json``
2. **Document parameters:** Add comments explaining parameter choices
3. **Test after loading:** Verify loaded pipeline produces expected results
4. **Commit to version control:** Include pipeline JSON in git repositories
5. **Archive with data:** Store pipeline configuration alongside processed data

Batch Processing
----------------

For processing multiple images efficiently, consider using :class:`ImagePipelineBatch`
which extends ImagePipeline with multiprocessing capabilities:

.. code-block:: python

    from phenotypic import ImagePipelineBatch, ImageSet
    
    # Create batch pipeline with parallel processing
    batch_pipeline = ImagePipelineBatch(
        ops=[GaussianBlur(), OtsuDetector()],
        meas=[MeasureShape(), MeasureIntensity()],
        njobs=4,  # Use 4 worker processes
        verbose=True
    )
    
    # Load image set
    image_set = ImageSet.from_directory('plates/')
    
    # Process in parallel
    results = batch_pipeline.apply_and_measure(image_set)

See :class:`ImagePipelineBatch` documentation for details on parallel processing,
memory management, and progress tracking for large-scale analyses.

Comparison & Utility
--------------------

The ImagePipeline class provides standard comparison and utility methods:

**String Representation:**

.. code-block:: python

    pipeline = ImagePipeline(
        ops=[OtsuDetector()],
        meas=[MeasureShape()]
    )
    
    print(pipeline)
    # Output: ImagePipeline(ops=1, meas=1, benchmark=False)
    
    print(repr(pipeline))
    # Output: <ImagePipeline: 1 operations, 1 measurements>

**Accessing Operations and Measurements:**

.. code-block:: python

    # Access operations dictionary
    ops_dict = pipeline._ops
    print(ops_dict.keys())  # ['OtsuDetector']
    
    # Access measurements dictionary
    meas_dict = pipeline._meas
    print(meas_dict.keys())  # ['MeasureShape']
    
    # Modify individual operation parameters
    pipeline._ops['OtsuDetector'].threshold_multiplier = 1.2

.. note::
   Direct modification of ``_ops`` and ``_meas`` dictionaries is possible but not
   recommended. Use ``set_ops()`` and ``set_meas()`` methods instead to ensure
   proper validation and naming.

