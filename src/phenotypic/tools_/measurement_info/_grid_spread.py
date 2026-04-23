"""Grid section spatial spread measurements."""

from phenotypic.abc_._measurement_info import MeasurementInfo


class GRID_SPREAD(MeasurementInfo):
    """Grid section spatial spread measurements.

    Provides measurements for evaluating spatial distribution of colonies within
    grid sections of arrayed microbial assays.
    """

    @classmethod
    def category(cls):
        return "GridSpread"

    OBJECT_SPREAD = (
        "ObjectSpread",
        "Sum of squared pairwise Euclidean distances between all unique colony pairs within a grid section. Quantifies spatial dispersion of colonies in a grid cell. Higher values indicate greater spread from the section center, suggesting over-segmentation, multi-detections, or colonies growing beyond expected boundaries. Used to identify problematic grid sections requiring refinement or quality review.",
    )
