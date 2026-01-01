"""Grid search utilities for pipeline parameter tuning and architecture comparison.

This module provides functions to perform parameter grid searches on ImagePipelines,
with multiple execution backends (joblib or submitit) and directory-based output with
interactive HTML viewer.

## Quick Start

### Single Pipeline Grid Search
For exploring parameter combinations of a single pipeline::

    from phenotypic import Image
    from phenotypic.enhance import GaussianBlur
    from phenotypic.util import PipelineGridSearchBase

    image = Image.imread('colony_plate.jpg')
    pipe_cfgs = [(GaussianBlur(sigma=1.0), {"sigma": [1.0, 2.0, 3.0]})]

    configs = PipelineGridSearchBase(
        image=image,
        pipe_cfgs=pipe_cfgs,
        output_dir="./grid_results",
        n_jobs=-1
    )

### Multi-Pipeline Grid Search
For comparing different pipeline architectures::

    from phenotypic.util import MultiPipelineGridSearch

    pipeline_configs = [
        {
            "name": "GaussianBlur_Otsu",
            "pipe_cfgs": [
                (GaussianBlur(sigma=1.0), {"sigma": [1.0, 2.0]}),
                (OtsuDetector(), {})
            ]
        },
        {
            "name": "MedianFilter_Otsu",
            "pipe_cfgs": [
                (MedianFilter(size=3), {"size": [3, 5]}),
                (OtsuDetector(), {})
            ]
        }
    ]

    configs = MultiPipelineGridSearch(
        image=image,
        pipeline_configs=pipeline_configs,
        output_dir="./multi_results",
        n_jobs=-1
    )

### Cluster Execution (Submitit Backend)
For submitting to SLURM clusters::

    configs = PipelineGridSearchBase(
        image=image,
        pipe_cfgs=pipe_cfgs,
        output_dir="./cluster_results",
        backend="submitit",
        slurm_params={"slurm_partition": "gpu", "mem_gb": 32}
    )

## Key Features

- **Directory-Based Output**: Results organized in individual subdirectories per pipeline
- **Interactive HTML Viewer**: Browse results with sidebar navigation and config display
- **Multiple Backends**: Choose between local (joblib) or cluster (submitit) execution
- **Memory Efficient**: Automatic cleanup and optimized array extraction
- **Shared Prefix Optimization**: MultiPipelineGridSearch reuses common preprocessing steps
- **Adaptive Batching**: Automatic memory management for large grid searches
- **Progress Tracking**: Terminal and Jupyter-compatible progress bars with ETA

## Output Structure

All grid search functions save results to organized directory structure::

    output_dir/
    ├── manifest.json           # Maps pipeline codes to configs + metadata
    ├── original/
    │   ├── rgb.tiff
    │   └── gray.tiff
    ├── pipeline_001/
    │   ├── rgb.tiff
    │   ├── gray.tiff
    │   ├── enh_gray.tiff
    │   ├── objmask.tiff
    │   └── objmap.tiff
    ├── pipeline_002/
    │   └── ...
    ├── thumbnails/             # Generated for HTML viewer
    │   └── ...
    └── viewer.html             # Interactive HTML viewer

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

1. **Enable adaptive batching (default):**
   ```python
   configs = MultiPipelineGridSearch(
       ...,
       adaptive_batching=True,
       memory_limit_gb=8.0,  # Set conservative limit
       n_jobs=4  # Reduce parallelism if needed
   )
   ```

2. **Reduce parallel workers to fit in memory:**
   - With 4096×4096 images: ~560 MB per pipeline
   - With n_jobs=4: ~2.2 GB peak memory
   - With n_jobs=-1 on 16 cores: ~9 GB peak memory (may OOM on 16 GB systems)

3. **Use submitit backend for cluster execution:**
   - Ideal for processing hundreds of pipelines across many jobs
   - Each job runs independently with its own memory allocation

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
