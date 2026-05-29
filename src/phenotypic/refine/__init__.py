"""Mask refinement for detected fungal colonies.

Post-detection operations that clean up binary masks by removing artifacts, fixing gaps,
and normalizing colony footprints across grid cells. Tools cover morphological operations
(opening, closing, erosion, dilation, gradient), hole filling, circularity checks, size
filtering, border exclusion, center deviation reduction, tophat-based brightening,
oversized-object capping, residual-based outlier removal, skeletonization, thinning,
and spatial merging of fragmented detections.

Merging operations (TransitiveDistanceMerger, NearestNeighborMerger, SmallToLargeMerger)
address fragmented colony detections from uneven illumination or heterogeneous pigmentation
by merging nearby detections based on spatial proximity and size thresholds.
"""

from ._trim_asymmetry import TrimAsymmetry
from ._remove_border_objects import RemoveBorderObjects
from ._keep_nearest_center import KeepNearestCenter
from ._circularity_modifier import LowCircularityRemover
from ._gmm_core_extractor import GMMCoreExtractor
from ._grid_alignment_refiner import GridAlignmentRefiner
from ._sine_alignment_refiner import SineAlignmentRefiner
from ._grid_oversized_object_remover import GridOversizedObjectRemover
from ._manual_refine import ManualRefine
from ._mask_closing import MaskClosing
from ._mask_dilation import MaskDilator
from ._mask_erosion import MaskErosion
from ._mask_fill import MaskFill
from ._mask_gradient import MaskGradient
from ._mask_opening import MaskOpening
from ._reduce_multiple_grid_objects import ReduceSectionsByLine
from ._nearest_neighbor_merger import NearestNeighborMerger
from ._residual_outlier_remover import ResidualOutlierRemover
from ._skeletonize import Skeletonize
from ._small_object_remover import SmallObjectRemover
from ._small_to_large_merger import SmallToLargeMerger
from ._thinning import Thinning
from ._transitive_distance_merger import TransitiveDistanceMerger
from ._white_tophat_modifier import WhiteTophat
from ._keep_section_largest import KeepSectionLargest
from ._separate_objects import SeparateObjects

__all__ = [
    "TrimAsymmetry",
    "RemoveBorderObjects",
    "KeepNearestCenter",
    "GMMCoreExtractor",
    "GridAlignmentRefiner",
    "GridOversizedObjectRemover",
    "LowCircularityRemover",
    "ManualRefine",
    "MaskClosing",
    "MaskDilator",
    "MaskErosion",
    "MaskFill",
    "MaskGradient",
    "MaskOpening",
    "ReduceSectionsByLine",
    "NearestNeighborMerger",
    "ResidualOutlierRemover",
    "SeparateObjects",
    "SineAlignmentRefiner",
    "Skeletonize",
    "SmallObjectRemover",
    "SmallToLargeMerger",
    "Thinning",
    "TransitiveDistanceMerger",
    "WhiteTophat",
    "KeepSectionLargest"
]
