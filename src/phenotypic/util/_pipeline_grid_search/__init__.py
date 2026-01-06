"""Pipeline grid search utilities for parameter optimization and comparison.

This module provides classes for executing systematic parameter grid searches
across multiple image processing pipelines, with support for both distributed
computing (via submitit/SLURM) and local parallel processing (via joblib).

**Public API:**

- `PipeGridSearch`: Recommended public interface for local parallel grid search
  using joblib with automatic memory-aware job scaling.

**Advanced/Specialized:**

- `PipelineGridSearchBase`: Abstract base class for custom grid search implementations.
- `PipeGridSearchJoblib`: Concrete joblib-based implementation (use via PipeGridSearch).
- `PipeGridSearchSubmitit`: For distributed HPC/SLURM execution on compute clusters.

**Typical Usage:**

.. code-block:: python

    from phenotypic.util._pipeline_grid_search import PipeGridSearch
    from phenotypic.enhance import GaussianBlur, CLAHE
    from phenotypic.detect import OtsuDetector
    from phenotypic import GridImage

    # Define parameter grid
    pipe_cfgs = {
        "Detection": [
            (GaussianBlur(), {"sigma": [1.0, 2.0]}),
            (CLAHE(), {"clip_limit": [1.5, 2.0]}),
            (OtsuDetector(), {"ignore_zeros": [True, False]}),
        ]
    }

    # Create grid search
    gs = PipeGridSearch(
        pipe_cfgs=pipe_cfgs,
        output_dir="/path/to/results",
        data2save={"enh_gray", "objmask"}
    )

    # Run with auto memory-aware job scaling
    image = GridImage.imread("plate.jpg", nrows=8, ncols=12)
    gs.process(image, njobs=-1)  # Auto-scale based on memory
"""

from ._pipe_grid_search_base import PipelineGridSearchBase
from ._pipe_grid_search_submitit import PipeGridSearchSubmitit
from ._pipe_grid_search_joblib import PipeGridSearchJoblib
from ._pipe_grid_search import PipeGridSearch

__all__ = [
    'PipeGridSearch',  # Public API (recommended)
    'PipelineGridSearchBase',  # Abstract base
    'PipeGridSearchJoblib',  # Joblib implementation (used by PipeGridSearch)
    'PipeGridSearchSubmitit',  # SLURM implementation
]
