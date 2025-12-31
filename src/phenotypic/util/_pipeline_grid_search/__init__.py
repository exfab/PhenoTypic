"""Grid search utilities for pipeline parameter tuning and architecture comparison.

This module provides functions to perform parameter grid searches on ImagePipelines,
with multiple execution backends (joblib or submitit) and visualization options (napari or TIFF).

## Quick Start

### Interactive Exploration (Napari Mode - Default)
For exploring parameter combinations with visual feedback::

    from phenotypic import Image
    from phenotypic.enhance import GaussianBlur
    from phenotypic.util import PipelineGridSearch

    image = Image.imread('colony_plate.jpg')
    ops = [(GaussianBlur(sigma=1.0), {"sigma": [1.0, 2.0, 3.0]})]

    viewer, configs = PipelineGridSearch(image=image, ops=ops, n_jobs=-1)

### Batch Processing (TIFF Mode - Memory Efficient)
For large grid searches without visualization overhead::

    configs = PipelineGridSearch(
        image=image,
        ops=ops,
        save_tiff_dir="./grid_results",
        create_trial_view=True,
        n_jobs=-1
    )

### Cluster Execution (Submitit Backend)
For submitting to SLURM clusters::

    configs = PipelineGridSearch(
        image=image,
        ops=ops,
        backend="submitit",
        slurm_params={"slurm_partition": "gpu", "mem_gb": 32},
        save_tiff_dir="./cluster_results"
    )

## Key Features

- **Multiple Backends**: Choose between local (joblib) or cluster (submitit) execution
- **Memory Efficient**: TIFF mode achieves 7-13× memory reduction by eliminating napari
- **HTML Reports**: Generate visual quality control pages with thumbnails
- **Shared Prefix Optimization**: Automatically optimize MultiPipelineGridSearch (enabled by default)
- **Automatic Memory Management**: Garbage collection and array cleanup after each batch
- **Progress Tracking**: Terminal and Jupyter-compatible progress bars with ETA

## Progress Tracking

All grid search functions provide automatic progress bars:
- **Terminal mode**: Standard tqdm progress bars with ETA
- **Jupyter mode**: Interactive widget-based progress bars (requires ipywidgets)
- **Multi-level operations**: Nested progress bars show batch → group → pipeline progress

Progress bars are automatically enabled for:
- Batch processing (MultiPipelineGridSearch with adaptive_batching=True)
- Trie group iteration (optimize_shared_prefixes=True)
- Parallel pipeline execution (when n_jobs != 1)

## Memory Optimization for Large Grids

When processing large numbers of pipelines with memory-intensive operations (e.g., BM3D):

1. **Use TIFF mode instead of napari viewer:**
   ```python
   configs = MultiPipelineGridSearch(
       ...,
       save_tiff_dir="./results",
       create_trial_view=True,
       n_jobs=4,  # Reduce parallelism
       memory_limit_gb=8.0  # Set conservative limit
   )
   ```

2. **Reduce parallel workers to fit in memory:**
   - With 4096×4096 images: ~560 MB per pipeline
   - With n_jobs=4: ~2.2 GB peak memory
   - With n_jobs=-1 on 16 cores: ~9 GB peak memory (may OOM on 16 GB systems)

3. **Use submitit backend for cluster execution:**
   - Auto-disables napari viewer (cannot display on clusters)
   - Requires save_tiff_dir parameter
   - Ideal for processing hundreds of pipelines across many jobs

## Notes

- On macOS, you may see a harmless "MallocStackLogging" warning from the system malloc library.
  This warning does not affect functionality and can be safely ignored.
"""

from ._multi_pipeline import MultiPipelineGridSearch
from ._single_pipeline import PipelineGridSearch

__all__ = [
    "PipelineGridSearch",
    "MultiPipelineGridSearch",
]
