from __future__ import annotations

import importlib.util
import os
from typing import Tuple, TYPE_CHECKING
from pathlib import Path
from joblib import Parallel, delayed

from ._pipe_grid_search_submitit import PipeGridSearchSubmitit

# Check for optional psutil dependency
HAS_PSUTIL = importlib.util.find_spec("psutil") is not None

if HAS_PSUTIL:
    import psutil

if TYPE_CHECKING:
    from phenotypic import Image
    from phenotypic.tools_.typing_ import GridSearchSaveData


class PipeGridSearchJoblib(PipeGridSearchSubmitit):
    """Parallel grid search using joblib with automatic memory-aware job scaling.
    
    This class extends PipelineGridSearchBase to provide local parallel processing
    of parameter grid searches using joblib's Parallel backend. It automatically
    calculates the optimal number of parallel jobs based on available system memory
    and image size, with seamless support for both terminal and Jupyter environments.
    
    The implementation is designed for arrayed microbial colony phenotyping on
    solid media agar, where images are processed through multiple parameter
    combinations to identify optimal detection and measurement settings.
    
    Args:
        pipe_cfgs: Dictionary of pipeline configurations (see parent class).
        output_dir: Directory for results (see parent class).
        data2save: Layers to save (see parent class).
    
    Attributes:
        pipe_cfgs: Pipeline configurations.
        output_dir: Output directory path.
        data2save: Image data layers to persist.
    
    Examples:
        >>> from phenotypic.util._pipeline_grid_search import PipeGridSearchJoblib
        >>> from phenotypic.enhance import GaussianBlur, CLAHE
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic import GridImage
        >>>
        >>> # Setup grid search configuration
        >>> pipe_cfgs = {
        ...     "DetectionPipeline": [
        ...         (GaussianBlur(), {"sigma": [1.0, 2.0, 3.0]}),
        ...         (CLAHE(), {"clip_limit": [1.0, 2.0]}),
        ...         (OtsuDetector(), {"ignore_zeros": [True, False]}),
        ...     ]
        ... }
        >>>
        >>> # Create search with output directory
        >>> gs = PipeGridSearchJoblib(
        ...     pipe_cfgs=pipe_cfgs,
        ...     output_dir="/path/to/results"
        ... )
        >>>
        >>> # Load test plate image (8x12 96-well plate)
        >>> image = GridImage.imread("plate_001.jpg", nrows=8, ncols=12)
        >>>
        >>> # Run parallel grid search with auto njobs based on memory
        >>> gs.process(image, njobs=-1)
        >>>
        >>> # Or specify explicit job count
        >>> gs.process(image, njobs=4)
    """

    def process(self, image: Image, njobs: int = -1) -> None:
        """Execute grid search with parallel pipeline processing.
        
        Orchestrates the complete grid search workflow: saves the original image,
        creates output directories, automatically calculates optimal parallel job
        count based on available memory (if njobs=-1), and processes all parameter
        combinations in parallel using joblib.
        
        The method detects the execution environment (terminal vs Jupyter) and
        adjusts progress reporting accordingly for optimal user experience.
        
        Args:
            image: Input Image or GridImage to process through all pipeline
                configurations. For plate-based experiments (e.g., 96-well
                screening), pass a GridImage with grid dimensions matching your
                plate format (nrows=8, ncols=12).
            njobs: Number of parallel jobs to use. If -1 (default), automatically
                calculates based on available memory and estimated image processing
                shape. Positive integers specify exact job count. Typical values:
                
                - -1: Auto-scale based on memory (aggressive: 95% RAM usage)
                - 1: Serial processing (useful for debugging)
                - 4: Conservative parallelism on 8-core system
                - -1: Maximum parallelism (all cores, memory-limited)
        
        Raises:
            ImportError: If njobs=-1 and psutil is not installed (required for
                automatic memory estimation).
            ValueError: If output directory does not exist or is invalid.
        
        Examples:
            >>> # Auto-scale to available memory (Jupyter-safe)
            >>> gs.process(image, njobs=-1)
            >>>
            >>> # Explicit parallel jobs (useful for cluster submissions)
            >>> gs.process(image, njobs=4)
            >>>
            >>> # Serial processing (debugging parameter combinations)
            >>> gs.process(image, njobs=1)
        """
        import logging

        logger = logging.getLogger(__name__)

        # Save original image and create directory structure
        logger.info("Preparing image and output directories...")
        self._prep_image(image)

        # Calculate optimal njobs if requested
        if njobs == -1:
            logger.info("Calculating optimal njobs based on image memory...")
            estimated_mem = self._estimate_image_memory(image)
            njobs = self._calculate_njobs_from_memory(estimated_mem)
            logger.info("Auto-calculated njobs=%d", njobs)
        else:
            logger.info("Using user-specified njobs=%d", njobs)

        # Collect all pipeline subdirectories
        pipe_dirs = []
        for pipe_config_dir in self.data_dir.iterdir():
            if pipe_config_dir.is_dir():
                for pipe_subdir in pipe_config_dir.iterdir():
                    if pipe_subdir.is_dir():
                        pipe_dirs.append(pipe_subdir)

        if not pipe_dirs:
            logger.warning("No pipeline directories found in %s", self.data_dir)
            return

        logger.info("Processing %d pipeline configurations with %d jobs...",
                    len(pipe_dirs), njobs)

        # Get backend and verbosity for environment
        backend, verbosity = self._get_progress_backend()

        # Execute parallel processing
        Parallel(
                n_jobs=njobs,
                backend=backend,
                verbose=verbosity
        )(
                delayed(PipelineGridSearchBase._process_single_pipe_dir)(
                        pipe_dir,
                        self._image_pkl_path,
                        self.data2save
                )
                for pipe_dir in pipe_dirs
        )

        logger.info("Grid search complete. Results saved to %s", self.output_dir)

    def _estimate_image_memory(self, image: Image) -> int:
        """Estimate memory shape of a single image processing job.
        
        Calculates the expected memory usage for processing one image through
        the entire pipeline. This includes the original image data plus overhead
        from intermediate arrays created during enhancement, detection, and
        measurement operations.
        
        Args:
            image: Input image to estimate memory for.
        
        Returns:
            Estimated memory usage in bytes.
        
        Examples:
            >>> est_bytes = gs._estimate_image_memory(image)
            >>> est_mb = est_bytes / (1024 ** 2)
            >>> print(f"Estimated memory per job: {est_mb:.1f} MB")
        """
        # Get base image size (RGB if available, otherwise grayscale)
        if not image.rgb.isempty():
            base_size = image.rgb[:].nbytes
        else:
            base_size = image.gray[:].nbytes

        # Apply multiplier for pipeline overhead
        # 5x accounts for: enhanced grayscale, masks, labels, intermediate
        # arrays during enhancement/detection, measurement data structures
        pipeline_multiplier = 5
        estimated_total = base_size * pipeline_multiplier

        return estimated_total

    def _calculate_njobs_from_memory(self, estimated_job_mem: int) -> int:
        """Calculate optimal parallel jobs based on available system memory.
        
        Determines the maximum number of parallel jobs that can run without
        exceeding 95% of available system memory. This aggressive scaling assumes
        modern systems with sufficient RAM, while gracefully handling memory-
        constrained environments.
        
        Falls back to CPU core count if psutil is unavailable. Always maintains
        a minimum of 1 job.
        
        Args:
            estimated_job_mem: Expected memory usage per job in bytes.
        
        Returns:
            Recommended number of parallel jobs to use.
        
        Raises:
            ImportError: If njobs=-1 was requested but psutil is not installed.
        
        Examples:
            >>> est_mem = gs._estimate_image_memory(image)
            >>> njobs = gs._calculate_njobs_from_memory(est_mem)
            >>> print(f"Using {njobs} parallel jobs")
        """
        cpu_count = os.cpu_count() or 1

        if not HAS_PSUTIL:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                    "psutil not installed. Falling back to CPU core count (%d jobs). "
                    "Install psutil for memory-aware job scaling.", cpu_count
            )
            return cpu_count

        # Get available system memory
        available_mem = psutil.virtual_memory().available

        # Use 95% of available memory (aggressive approach)
        usable_mem = int(available_mem * 0.95)

        # Calculate max jobs based on estimated memory per job
        max_jobs_from_memory = max(1, int(usable_mem / estimated_job_mem))

        # Limit to CPU core count (no benefit in oversubscribing)
        optimal_jobs = min(max_jobs_from_memory, cpu_count)

        # Ensure at least 1 job
        optimal_jobs = max(1, optimal_jobs)

        # Log decision for transparency
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
                "Memory-aware job scaling: %d cores available, %.1f GB available, "
                "%.1f GB per job estimate → using %d parallel jobs",
                cpu_count,
                available_mem / (1024 ** 3),
                estimated_job_mem / (1024 ** 3),
                optimal_jobs
        )

        return optimal_jobs

    def _get_progress_backend(self) -> Tuple[str, int]:
        """Detect execution environment and return joblib backend configuration.
        
        Distinguishes between terminal and Jupyter environments to select
        appropriate joblib backend and verbosity level for progress reporting.
        Terminal mode uses high verbosity for real-time job tracking, while
        Jupyter mode uses lower verbosity to avoid excessive output in notebooks.
        
        Returns:
            Tuple of (backend_name, verbosity_level) where:
            
            - backend_name: 'loky' for spawned processes (recommended)
            - verbosity_level: 10 for terminal (high progress detail),
              5 for Jupyter (compatible with notebook rendering)
        
        Examples:
            >>> backend, verbosity = gs._get_progress_backend()
            >>> print(f"Using {backend} backend with verbosity={verbosity}")
        """
        # Detect Jupyter environment
        try:
            from IPython import get_ipython

            ipython = get_ipython()
            if ipython is not None and 'IPKernelApp' in ipython.config:
                # Running in Jupyter notebook/JupyterLab
                return ('loky', 5)
        except (ImportError, AttributeError):
            pass

        # Terminal environment or IPython detection failed
        return ('loky', 10)
