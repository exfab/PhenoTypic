"""Radial expansion measurements for filamentous fungal colonies."""

from ._measurement_info import MeasurementInfo


class RADIAL_EXPANSION(MeasurementInfo):
    """Radial expansion measurements for filamentous fungal colonies.

    Quantifies branching morphology by decomposing colony structure into a
    dense core and peripheral branches via PELT changepoint detection on
    radial density profiles, followed by skeleton-based branch tracing.
    Includes runner detection for identifying anomalously long branches.
    """

    @classmethod
    def category(cls) -> str:
        return "RadialExpansion"

    ROBUST_MEAN_RADIUS = (
        "RobustMeanRadius",
        "Trimmed mean of branch path lengths, excluding detected runner. "
        "Represents the typical radial expansion distance of hyphae from the "
        "core boundary, robust to single-runner outliers.",
    )
    MEAN_RADIUS = (
        "MeanRadius",
        "Arithmetic mean of all branch path lengths from core boundary to "
        "branch tips. Includes runner if present. Reflects overall average "
        "colony reach.",
    )
    MEDIAN_RADIUS = (
        "MedianRadius",
        "Median branch path length. Robust to outliers and provides a "
        "typical expansion distance.",
    )
    NUM_BRANCHES = (
        "NumBranches",
        "Number of skeleton branches extending from the core boundary to "
        "peripheral tips. Higher counts indicate denser branching morphology.",
    )
    MAX_BRANCH_LENGTH = (
        "MaxBranchLength",
        "Length of the longest branch path in pixels. When a runner is "
        "detected, this equals RunnerLength.",
    )
    RUNNER_LENGTH = (
        "RunnerLength",
        "Path length of the detected runner branch (the single longest "
        "outlier). NaN if no runner detected.",
    )
    RUNNER_DETECTED = (
        "RunnerDetected",
        "Boolean flag (0 or 1) indicating whether an outlier runner branch "
        "was detected for this colony.",
    )
    CORE_RADIUS = (
        "CoreRadius",
        "Radius of the dense colony core as determined by PELT changepoint "
        "detection on the radial density profile. Pixels within this radius "
        "of the intensity-weighted centroid are excluded from skeleton analysis.",
    )
