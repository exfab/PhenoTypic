"""Prefab pipelines for fungal colony plate processing.

Ready-to-run chains of enhancement, detection, refinement, and measurement steps
tuned for common agar plate scenarios. Includes watershed-heavy pipelines for
clustered colonies, Otsu-based pipelines for clean backgrounds, grid-section pipelines
for tiled inputs, and grid-aware Gitter-style processing for dense arrays.
"""

from ._heavy_watershed_pipeline import HeavyWatershedPipeline
from ._heavy_otsu_pipeline import HeavyOtsuPipeline
from ._grid_section_pipeline import GridSectionPipeline
from ._heavy_gitter_pipeline import HeavyGitterPipeline

__all__ = [
    "HeavyWatershedPipeline",
    "HeavyOtsuPipeline",
    "GridSectionPipeline",
    "HeavyGitterPipeline",
]
