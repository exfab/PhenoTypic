# MultiPipelineGridSearch Data Flow Diagrams

This directory contains data flow diagrams for the `MultiPipelineGridSearch` function,
which provides advanced grid search capabilities across multiple pipeline configurations
with memory optimization and parallel processing.

## Diagram Files

### 1. `multi_pipeline_grid_search_data_flow.md`

**High-level overview diagram** showing the main decision points and processing paths:

- Input validation and backend selection
- Choice between optimized (trie-based) vs non-optimized execution
- Result processing modes (napari viewer vs TIFF saving)
- Final outputs

### 2. `multi_pipeline_grid_search_detailed_data_flow.md`

**Detailed data flow diagram** showing:

- All key functions and data transformations
- Memory management and batching logic
- Parallel execution strategies
- Result extraction and storage processes

### 3. `multi_pipeline_grid_search_overview.md`

**Architectural overview** showing:

- Input/output relationships
- Key decision points
- Processing strategies and their trade-offs
- Memory management integration

## Key Concepts Illustrated

### Execution Strategies

1. **Trie-based Optimization** (`optimize_shared_prefixes=True`)
    - Expands all pipeline configurations to concrete parameter combinations
    - Groups pipelines by shared operation prefixes using a trie structure
    - Executes shared prefixes only once, reusing results for divergent branches
    - Significantly reduces computation time for pipelines with common operations

2. **Linear Processing** (`optimize_shared_prefixes=False`)
    - Processes each pipeline configuration independently
    - Generates parameter combinations per pipeline
    - Executes each pipeline's grid search in parallel
    - Original behavior, useful for debugging or comparison

### Memory Management

- **Adaptive Batching**: Automatically calculates optimal batch sizes based on:
    - Available system memory
    - Estimated memory per pipeline
    - Number of parallel jobs
- **Memory Monitoring**: Tracks memory usage throughout execution
- **Garbage Collection**: Explicit cleanup after each batch/pipeline

### Backend Options

1. **Joblib Backend** (local processing)
    - Uses multiprocessing/threading for parallel execution
    - Supports both napari viewer and TIFF saving modes
    - Can use trie optimization

2. **Submitit Backend** (SLURM cluster)
    - Submits jobs to SLURM cluster for distributed processing
    - Only supports TIFF saving mode (no interactive viewer)
    - Disables trie optimization (jobs are already parallelized)

### Output Modes

1. **Napari Viewer Mode** (`save_tiff_dir=None`)
    - Creates interactive napari viewer with all results as layers
    - Returns `(viewer, configs_dict)` tuple
    - Best for exploratory analysis and visual QC

2. **TIFF Saving Mode** (`save_tiff_dir` specified)
    - Saves all result layers as TIFF files to specified directory
    - Memory efficient (3-13× reduction)
    - Can generate HTML overview page
    - Returns only `configs_dict`
    - Required for cluster processing

## Usage Examples

### Basic Usage

```python
from phenotypic.util import MultiPipelineGridSearch

# Compare different preprocessing strategies
pipeline_configs = [
    {
        "name": "Gaussian_Otsu",
        "pipe_cfgs": [
            (GaussianBlur(sigma=1.0), {"sigma": [1.0, 2.0]}),
            (OtsuDetector(), {})
        ]
    },
    {
        "name": "Median_Otsu",
        "pipe_cfgs": [
            (MedianFilter(size=3), {"size": [3, 5]}),
            (OtsuDetector(), {})
        ]
    }
]

# Interactive exploration
viewer, configs = MultiPipelineGridSearch(image, pipeline_configs)

# Batch processing for cluster
configs = MultiPipelineGridSearch(
    image, pipeline_configs,
    save_tiff_dir="./results",
    backend="submitit"
)
```

## Performance Optimizations

The diagrams illustrate several key optimizations:

1. **Shared Prefix Optimization**: Automatically detects and reuses computation for
   pipelines with common starting operations
2. **Adaptive Batching**: Prevents out-of-memory errors by processing pipelines in
   optimal-sized batches
3. **Memory Estimation**: Calculates memory requirements per pipeline to inform batching
   decisions
4. **Parallel Execution**: Utilizes multiple cores/jobs for parallel processing
5. **Memory Cleanup**: Explicit garbage collection and array deletion to minimize memory
   footprint

## Viewing the Diagrams

These diagrams are in Markdown format with Mermaid code blocks. You can view them by:

1. **GitHub**: Most GitHub repositories render Mermaid code blocks automatically
2. **VS Code**: Install the "Mermaid Preview" extension to preview diagrams
3. **Online**: Copy the Mermaid code block content to mermaid.live or other online
   viewers
4. **Documentation**: These diagrams can be included directly in Sphinx documentation
   using the `mermaid` directive
5. **Markdown Preview**: Any Markdown renderer that supports Mermaid (like GitHub,
   GitLab, etc.)

## Related Files

- **Implementation**: `src/phenotypic/util/_pipeline_grid_search/_multi_pipeline.py`
- **Helper Functions**: `src/phenotypic/util/_pipeline_grid_search/_shared.py`
- **Tests**: `tests/test_pipeline_grid_search_backends.py`
- **CLI Integration**: `src/phenotypic/phenotypicCLI.py`
