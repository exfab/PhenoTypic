"""Mask refinement for detected fungal colonies.

Post-detection operations that clean up binary masks by removing artifacts, fixing gaps,
and normalizing colony footprints across grid cells. Tools cover morphological operations
(opening, closing, erosion, dilation, gradient), hole filling, circularity checks, size
filtering, border exclusion, center deviation reduction, tophat-based brightening,
oversized-object capping, residual-based outlier removal, skeletonization, thinning,
and spatial merging of fragmented detections.

Merging operations (MergeFragmentChains, MergeNearestNeighbors, MergeSmallToLarge)
address fragmented colony detections from uneven illumination or heterogeneous pigmentation
by merging nearby detections based on spatial proximity and size thresholds.
"""

from ._trim_asymmetry import TrimAsymmetry
from ._remove_border_objects import RemoveBorderObjects
from ._keep_nearest_center import KeepNearestCenter
from ._remove_low_circularity import RemoveLowCircularity
from ._extract_colony_core import ExtractColonyCore
from ._grid_alignment_refiner import RefineGridAlignment
from ._refine_by_sine_fit import RefineBySineFit
from ._grid_oversized_object_remover import RemoveOversizedGridObjects
from ._manual_refine import ManualRefine
from ._mask_closing import MaskClosing
from ._mask_dilation import MaskDilation
from ._mask_erosion import MaskErosion
from ._mask_fill import MaskFill
from ._mask_gradient import MaskGradient
from ._mask_opening import MaskOpening
from ._reduce_multiple_grid_objects import ReduceSectionsByLine
from ._nearest_neighbor_merger import MergeNearestNeighbors
from ._remove_grid_outliers import RemoveGridOutliers
from ._skeletonize import Skeletonize
from ._small_object_remover import RemoveSmallObjects
from ._small_to_large_merger import MergeSmallToLarge
from ._thinning import Thinning
from ._merge_fragment_chains import MergeFragmentChains
from ._mask_white_tophat import MaskWhiteTophat
from ._keep_section_largest import KeepSectionLargest
from ._separate_objects import SeparateObjects
from ._remove_by_feature import RemoveByFeature

__all__ = [
    "TrimAsymmetry",
    "RemoveBorderObjects",
    "KeepNearestCenter",
    "ExtractColonyCore",
    "RefineGridAlignment",
    "RemoveOversizedGridObjects",
    "RemoveLowCircularity",
    "ManualRefine",
    "MaskClosing",
    "MaskDilation",
    "MaskErosion",
    "MaskFill",
    "MaskGradient",
    "MaskOpening",
    "ReduceSectionsByLine",
    "MergeNearestNeighbors",
    "RemoveGridOutliers",
    "SeparateObjects",
    "RefineBySineFit",
    "Skeletonize",
    "RemoveSmallObjects",
    "MergeSmallToLarge",
    "Thinning",
    "MergeFragmentChains",
    "MaskWhiteTophat",
    "KeepSectionLargest",
    "RemoveByFeature",
]


_LEGACY_OPERATION_NAMES = {
    "GridOversizedObjectRemover": RemoveOversizedGridObjects,
    "SmallObjectRemover": RemoveSmallObjects,
    "NearestNeighborMerger": MergeNearestNeighbors,
    "SmallToLargeMerger": MergeSmallToLarge,
    "GridAlignmentRefiner": RefineGridAlignment,
}


def __getattr__(name: str):
    """Resolve deprecated refine operation names without GUI duplicates."""
    operation = _LEGACY_OPERATION_NAMES.get(name)
    if operation is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import warnings

    warnings.warn(
        f"{name} is deprecated; use {operation.__name__} instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return operation
