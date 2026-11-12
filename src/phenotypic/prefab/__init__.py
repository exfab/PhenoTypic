"""Prefab pipelines for fungal colony plate processing.

Ready-to-run chains of enhancement, detection, refinement, and measurement steps
tuned for common agar plate scenarios. Includes watershed-heavy pipelines for
clustered colonies, Otsu-based pipelines for clean backgrounds, grid-section pipelines
for tiled inputs, grid-aware Gitter-style processing for dense arrays, and a
SAM2-based pipeline for bacterial colonies on pre-corrected images.
"""

from ._heavy_watershed_pipeline import HeavyWatershedPipeline
from ._heavy_otsu_pipeline import HeavyOtsuPipeline
from ._grid_section_pipeline import GridSectionPipeline
from ._heavy_round_peaks_pipeline import HeavyRoundPeaksPipeline
from ._round_peaks_pipeline import RoundPeaksPipeline
from ._filamentous_fungi_pipeline import FilamentousFungiPipeline
from ._spimager_pipeline import SpImagerPipeline
from ._bactersam_pipeline import BacterSamPipeline

__all__ = [
    "HeavyWatershedPipeline",
    "HeavyOtsuPipeline",
    "GridSectionPipeline",
    "HeavyRoundPeaksPipeline",
    "RoundPeaksPipeline",
    "FilamentousFungiPipeline",
    "SpImagerPipeline",
    "BacterSamPipeline",
]
