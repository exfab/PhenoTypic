"""Pipeline grid search utilities for parameter optimization and comparison.

.. deprecated::
    This module is deprecated. Use :mod:`phenotypic.sweep` instead for a
    simpler API based on :class:`~phenotypic.sweep.Sweep` objects::

        from phenotypic.sweep import Sweep, generate_sweep_manifest
        from phenotypic.enhance import GaussianBlur
        from phenotypic.detect import OtsuDetector

        config = [
            Sweep(GaussianBlur, sigma=(1.0, 2.0)),
            Sweep(OtsuDetector, ignore_zeros=(True, False)),
        ]
        manifest = generate_sweep_manifest(config)

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
