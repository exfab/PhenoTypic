from ._pipe_grid_search_joblib import PipeGridSearchJoblib


class PipeGridSearch(PipeGridSearchJoblib):
    """Public interface for systematic parameter grid search of image processing pipelines.

    PipeGridSearch provides exhaustive parameter combination testing for colony detection
    and measurement pipelines, with support for both local parallel processing (joblib)
    and distributed HPC/SLURM execution (submitit). This is the recommended public API
    for pipeline optimization in arrayed microbial colony phenotyping workflows.

    **Processing Methods:**

    **Local Processing (Joblib):**
    Use :meth:`process` for joblib-based parallel execution with automatic memory-aware job scaling.
    Suitable for workstations and small compute clusters. Automatically detects Jupyter vs
    terminal environments and adjusts progress reporting accordingly.

    **Distributed Processing (SLURM):**
    Use :meth:`~phenotypic.util._pipeline_grid_search.PipeGridSearchSubmitit.submitit`
    for HPC cluster execution via SLURM workload manager. Requires submitit dependency
    and appropriate SLURM configuration.

    **Key Features:**

    * **Memory-Aware Parallelism:** Automatically scales parallel jobs based on available
      system memory and estimated per-image processing shape (5× base image size).

    * **Hierarchical Results:** Organizes outputs in structured directory tree with
      reproducible JSON-serialized pipelines for each parameter combination.

    * **Environment Detection:** Adjusts progress reporting for optimal experience in
      Jupyter notebooks vs terminal execution.

    * **Selective Output:** Choose which image layers to persist (RGB, grayscale,
      detection matrix, object masks, labeled maps) to control disk usage.

    Args:
        pipe_cfgs (Dict[str, List[Tuple[ImageOperation, Dict[str, List[Any]]]]]):
            Dictionary mapping pipeline configuration names to operation lists. Each value
            is a list of ``(operation_instance, parameter_dict)`` tuples where:

            - ``operation_instance``: An :class:`~phenotypic.abc_.ImageOperation` subclass instance
            - ``parameter_dict``: Maps parameter names to lists of values to test

            Example:
                .. code-block:: python

                    {
                        "DetectionPipeline": [
                            (GaussianBlur(), {"sigma": [1.0, 1.5, 2.0]}),
                            (CLAHE(), {"clip_limit": [1.5, 2.0, 2.5]}),
                            (OtsuDetector(), {"ignore_zeros": [True, False]}),
                        ]
                    }

            The grid search generates all parameter combinations using
            :func:`itertools.product`, creating ``len(sigma) × len(clip_limit) × len(ignore_zeros)``
            unique pipeline configurations.

        output_dir (Path | str | None):
            Directory where results will be saved. Must exist and be writable. If ``None``,
            uses current working directory. Results are organized into ``output_dir/data/``
            with hierarchical structure per pipeline configuration.

        data2save (Set[str], optional):
            Image layers to persist to disk. Valid options:

            - ``"rgb"``: Original color image (TIFF, if available)
            - ``"gray"``: Grayscale luminance (TIFF)
            - ``"detect_mat"``: Detection matrix for processing (TIFF)
            - ``"objmask"``: Binary detection mask (PNG)
            - ``"objmap"``: Labeled object map (PNG)
            - ``"map2rgb"``: Object map rendered as RGB overlay (PNG)

            Defaults to all layers. Use subset to minimize disk usage for large batches.

    Raises:
        ValueError: If ``output_dir`` doesn't exist/is invalid, or pipeline configurations
            are malformed (invalid tuples, non-existent parameters, etc.).
        ImportError: If ``njobs=-1`` requested but ``psutil`` not installed (memory scaling).

    **Output Directory Structure:**

    .. code-block:: text

        output_dir/
        ├── OriginalImage.pkl              # Pickled original image
        ├── OriginalRGB.tif                # RGB backup (if input had RGB)
        ├── OriginalGray.tif               # Grayscale backup
        └── data/
            ├── PipelineName1/
            │   ├── PipelineName1_0/       # Parameter combination 0
            │   │   ├── PipelineName1_0.json  # Serialized pipeline
            │   │   ├── rgb.tiff          # (if in data2save)
            │   │   ├── objmask.png       # (if in data2save)
            │   │   └── PipelineName1_0_Image.pkl  # Processed result
            │   └── PipelineName1_1/       # Parameter combination 1
            │       └── ...
            └── PipelineName2/
                └── ...

    **Processing Methods:**

    **Local Parallel Processing:**

    Use :meth:`process` for joblib-based parallel execution on local hardware:

    .. code-block:: python

        from phenotypic.util._pipeline_grid_search import PipeGridSearch
        from phenotypic.enhance import GaussianBlur, CLAHE
        from phenotypic.detect import OtsuDetector
        from phenotypic import GridImage

        # Define parameter grid (4×3×2 = 24 combinations)
        pipe_cfgs = {
            "DetectionPipeline": [
                (GaussianBlur(), {"sigma": [1.0, 1.5, 2.0, 2.5]}),
                (CLAHE(), {"clip_limit": [1.5, 2.0, 2.5]}),
                (OtsuDetector(), {"ignore_zeros": [True, False]}),
            ]
        }

        # Create grid search
        gs = PipeGridSearch(
            pipe_cfgs=pipe_cfgs,
            output_dir="/path/to/results",
            data2save={"detect_mat", "objmask"}
        )

        # Load 96-well plate image
        image = GridImage.imread("plate_001.jpg", nrows=8, ncols=12)

        # Execute with auto memory-aware job scaling
        gs.process(image, njobs=-1)  # Auto-scale based on available RAM

        # Or specify explicit job count
        gs.process(image, njobs=4)   # Use exactly 4 parallel workers

    **Job Scaling Options:**

    - ``njobs=-1`` (default): Auto-calculate based on memory using 95% of available RAM
    - ``njobs=N`` (positive): Use exactly N parallel workers
    - ``njobs=1``: Serial processing (useful for debugging)

    **Distributed HPC Processing:**

    For SLURM clusters, use :class:`~phenotypic.util._pipeline_grid_search.PipeGridSearchSubmitit`:

    .. code-block:: python

        from phenotypic.util._pipeline_grid_search import PipeGridSearchSubmitit

        # Create SLURM-capable grid search
        gs_slurm = PipeGridSearchSubmitit(pipe_cfgs, output_dir="/cluster/results")

        # Configure SLURM job parameters
        slurm_config = {
            "slurm_job_name": "colony_grid_search",
            "slurm_time": "04:00:00",      # 4 hour time limit
            "slurm_mem": "8GB",            # Memory per task
            "slurm_cpus_per_task": 1,      # CPUs per task
            "slurm_partition": "gpu",      # Partition/queue
        }

        # Submit to SLURM scheduler
        gs_slurm.submitit(image, slurm_config)

    **Memory Estimation:**

    Local processing estimates per-pipeline memory as ``base_image_size × 5``, accounting for:
    detection matrix, detection masks, intermediate processing arrays, and measurement data.

    **Use Cases:**

    1. **Parameter Optimization:** Discover optimal detection settings for your imaging
       hardware and specimen preparation protocols.

    2. **Sensitivity Analysis:** Assess how detection accuracy varies across parameter
       ranges in colony phenotyping pipelines.

    3. **Pipeline Comparison:** Compare results across different operation combinations
       to select the best workflow for plate-based experiments.

    4. **Hardware Validation:** Test pipeline robustness across different cameras,
       lighting conditions, and plate formats.

    **Reproducibility:**

    Each parameter combination is saved as a self-contained JSON file, enabling
    independent re-execution:

    .. code-block:: python

        # Re-run a specific configuration
        import phenotypic as pht

        pipe = pht.ImagePipeline.from_json("results/data/Pipeline1/Pipeline1_5.json")
        result = pipe.apply(image)

    See Also:
        * :class:`~phenotypic.util._pipeline_grid_search.PipeGridSearchSubmitit`: For SLURM/HPC execution
        * :class:`~phenotypic.core._image_pipeline.ImagePipeline`: For sequential operation chaining
        * :class:`~phenotypic.GridImage`: For plate-based image analysis
        * :class:`~phenotypic.abc_.ImageOperation`: Base class for pipeline operations
    """

    pass
