"""Grid section spatial spread measurements."""

from ._measurement_info import Entry
from ._tiers import QualityInfo


class GRID_SPREAD(QualityInfo):
    """Quantify within-well colony dispersion using pairwise centroid distances.

    Compute the sum of squared pairwise Euclidean distances between all
    colony centroids in each grid section. High values indicate multiple
    dispersed objects within a single well -- a sign of over-segmentation,
    fragmented growth, or invasive spreading.
    """

    @classmethod
    def category(cls):
        return "GridSpread"

    OBJECT_SPREAD = Entry(
        "ObjectSpread",
        "Sum of squared pairwise Euclidean distances between all unique colony pairs within a grid section. Quantifies spatial dispersion of colonies in a grid cell. Higher values indicate greater spread from the section center, suggesting over-segmentation, multi-detections, or colonies growing beyond expected boundaries. Used to identify problematic grid sections requiring refinement or quality review.",
    )
