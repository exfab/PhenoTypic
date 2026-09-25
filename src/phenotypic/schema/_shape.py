"""The labels and descriptions of the shape measurements."""

from ._change_notes import SIZE_SHAPE_SPLIT_NOTE
from ._measurement_info import Entry
from ._tiers import PrimaryMeasure


class SHAPE(PrimaryMeasure):
    """Measure the form of each detected colony.

    Extract dimensionless and angular form descriptors -- circularity,
    compactness, solidity, extent, eccentricity, orientation -- plus the
    Feret caliper diameters and the colony's interior thickness (distance
    from its pixels to the nearest edge). Size magnitudes (area, perimeter,
    radii, axis lengths) live in :class:`SIZE`.
    """

    @classmethod
    def category(cls):
        return "Shape"

    @classmethod
    def change_note(cls) -> str:
        return SIZE_SHAPE_SPLIT_NOTE

    @classmethod
    def tier(cls) -> int:
        return 2  # default for form descriptors; Feret diameters override via Entry(tier=1)

    CIRCULARITY = Entry(
        "Circularity",
        r"Calculated as :math:`\frac{4\pi*\text{Area}}{\text{Perimeter}^2}`. Measures how closely a colony approximates a perfect circle (value = 1). Values < 1 indicate irregular colony morphology, which may result from genetic mutations, environmental stress, or mixed microbial populations on agar plates.",
    )
    MIN_FERET_DIAMETER = Entry(
        "MinFeretDiameter",
        "Minimum caliper diameter - the shortest distance between two parallel tangent lines touching opposite sides of the colony. Represents the narrowest dimension of the colony regardless of orientation. Useful for detecting elongated or irregular colony morphologies and measuring colony width.",
        tier=1,
    )
    MAX_FERET_DIAMETER = Entry(
        "MaxFeretDiameter",
        "Maximum caliper diameter - the longest distance between two parallel tangent lines touching opposite sides of the colony. Represents the maximum dimension of the colony regardless of orientation. Often exceeds major axis length for irregular shapes and helps quantify maximum colony extent.",
        tier=1,
    )
    ECCENTRICITY = Entry(
        "Eccentricity",
        "Measure of colony elongation, ranging from 0 (perfect circle) to 1 (highly elongated). Values near 0 indicate compact, radially symmetric growth typical of healthy bacterial colonies, while higher values may suggest directional growth, motility, or environmental gradients on the agar surface.",
    )
    SOLIDITY = Entry(
        "Solidity",
        "Ratio of actual colony area to its convex hull area (Area/ConvexArea). Values near 1 indicate compact, solid colonies with minimal indentations. Lower values (< 0.9) may indicate invasive growth, colony spreading, or the presence of clearing zones around colonies.",
    )
    EXTENT = Entry(
        "Extent",
        "Ratio of colony area to its bounding box area (ObjectArea/BboxArea). Measures how efficiently the colony fills its allocated space. Compact colonies have higher extent values, while spread-out or irregular colonies have lower values.",
    )
    COMPACTNESS = Entry(
        "Compactness",
        r"Calculated as :math:`\frac{\text{Perimeter}^2}{4\pi*\text{Area}}`. Inverse of circularity (ranges from 1 for perfect circles to higher values for irregular shapes). Measures colony shape complexity - compact, circular colonies have values near 1, while irregular or filamentous colonies have much higher values.",
    )
    ORIENTATION = Entry(
        "Orientation",
        "Angle (in radians) between the colony's major axis and the horizontal axis. Measures colony alignment and growth directionality. Random orientations are typical for most bacterial colonies, while consistent orientations may indicate environmental gradients or mechanical stresses during plating.",
    )
    MEAN_BOUNDARY_DIST = Entry(
        "MeanBoundaryDist",
        "Mean Euclidean distance from each colony pixel to the nearest background "
        "pixel, computed on the object in isolation. This is a measure of interior "
        "thickness, not a radius: for an ideal disk of radius R it equals "
        r":math:`R/3`. High values relative to Size_InscribedRadius indicate a "
        "compact, convex colony; low values indicate a thin or filamentous one.",
    )
    MEDIAN_BOUNDARY_DIST = Entry(
        "MedianBoundaryDist",
        "Median Euclidean distance from each colony pixel to the nearest background "
        "pixel, computed on the object in isolation. This is a measure of interior "
        "thickness, not a radius: for an ideal disk of radius R it equals "
        r":math:`R(1 - 1/\sqrt{2}) \approx 0.293R`. More robust to boundary "
        "raggedness than MeanBoundaryDist. See Size_InscribedRadius and "
        "Size_RobustMeanRadius for the colony's radial extent.",
    )


SHAPE.__doc__ = f"{SHAPE.__doc__}\n\n{SHAPE.change_note()}"
