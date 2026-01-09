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

from ._border_object_modifier import BorderObjectRemover
from ._center_deviation_reducer import CenterDeviationReducer
from ._circularity_modifier import LowCircularityRemover
from ._grid_oversized_object_remover import GridOversizedObjectRemover
from ._mask_closer import MaskCloser
from ._mask_dilator import MaskDilator
from ._mask_eroder import MaskEroder
from ._mask_fill import MaskFill
from ._mask_gradient import MaskGradient
from ._mask_opener import MaskOpener
from ._min_residual_error_reducer import ReduceMultipleGridObjects
from ._nearest_neighbor_merger import NearestNeighborMerger
from ._residual_outlier_remover import ResidualOutlierRemover
from ._skeletonize import Skeletonize
from ._small_object_modifier import SmallObjectRemover
from ._small_to_large_merger import SmallToLargeMerger
from ._thinning import Thinning
from ._transitive_distance_merger import TransitiveDistanceMerger
from ._white_tophat_modifier import WhiteTophat
from ._grid_section_largest import GridSectionLargest

__all__ = [
    "BorderObjectRemover",
    "CenterDeviationReducer",
    "GridOversizedObjectRemover",
    "LowCircularityRemover",
    "MaskCloser",
    "MaskDilator",
    "MaskEroder",
    "MaskFill",
    "MaskGradient",
    "MaskOpener",
    "ReduceMultipleGridObjects",
    "NearestNeighborMerger",
    "ResidualOutlierRemover",
    "Skeletonize",
    "SmallObjectRemover",
    "SmallToLargeMerger",
    "Thinning",
    "TransitiveDistanceMerger",
    "WhiteTophat",
    "GridSectionLargest"
]
