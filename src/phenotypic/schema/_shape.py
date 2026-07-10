"""The labels and descriptions of the shape measurements."""

from ._measurement_info import Entry
from ._tiers import PrimaryMeasure


class SHAPE(PrimaryMeasure):
    """Measure comprehensive morphological characteristics of detected colonies.

    Extract geometric metrics from each colony shape: area, perimeter,
    circularity, convex hull properties, width-based measures, Feret
    diameters, eccentricity, and best-fit ellipse parameters. The output
    DataFrame provides a full morphological profile for phenotypic
    classification and growth-pattern analysis.
    """

    @classmethod
    def category(cls):
        return "Shape"

    @classmethod
    def tier(cls) -> int:
        return 2  # default for form descriptors; size-magnitude members override via Entry(tier=1)

    AREA = Entry(
        "Area",
        "Total number of pixels occupied by the microbial colony. Represents colony biomass and growth extent on agar plates. Larger areas typically indicate more robust growth or longer incubation times.",
        bio_desc=(
            "Projected 2D footprint of the colony in pixels — a common proxy "
            "for colony size and overall growth in arrayed plate assays. With "
            "matched imaging and incubation, larger area generally reflects "
            "greater proliferation or spreading; it captures only the 2D "
            "footprint, not colony height or cell density."
        ),
        image="shape/area.png",
        tier=1,
    )
    PERIMETER = Entry(
        "Perimeter",
        "Total length of the colony's outer boundary in pixels. Measures colony edge complexity and surface irregularity. Smooth, circular colonies have shorter perimeters relative to their area compared to irregular or filamentous colonies.",
        tier=1,
    )
    CIRCULARITY = Entry(
        "Circularity",
        r"Calculated as :math:`\frac{4\pi*\text{Area}}{\text{Perimeter}^2}`. Measures how closely a colony approximates a perfect circle (value = 1). Values < 1 indicate irregular colony morphology, which may result from genetic mutations, environmental stress, or mixed microbial populations on agar plates.",
    )
    CONVEX_AREA = Entry(
        "ConvexArea",
        'Area of the smallest convex polygon that completely contains the colony. Represents the colony\'s "filled-in" appearance if all indentations and holes were removed. Useful for detecting colony spreading patterns or invasive growth characteristics.',
        tier=1,
    )
    MEDIAN_BOUNDARY_DIST = Entry(
        "MedianBoundaryDist",
        "Median Euclidean distance from each colony pixel to the nearest background "
        "pixel, computed on the object in isolation. This is a measure of interior "
        "thickness, not a radius: for an ideal disk of radius R it equals "
        r":math:`R(1 - 1/\sqrt{2}) \approx 0.293R`. More robust to boundary raggedness "
        "than MeanBoundaryDist. See InscribedRadius and RobustMeanRadius for the "
        "colony's actual radial extent.",
        tier=1,
    )
    MEAN_BOUNDARY_DIST = Entry(
        "MeanBoundaryDist",
        "Mean Euclidean distance from each colony pixel to the nearest background "
        "pixel, computed on the object in isolation. This is a measure of interior "
        "thickness, not a radius: for an ideal disk of radius R it equals "
        r":math:`R/3`. High values relative to InscribedRadius indicate a compact, "
        "convex colony; low values indicate a thin or filamentous one.",
        tier=1,
    )
    INSCRIBED_RADIUS = Entry(
        "InscribedRadius",
        "Radius of the largest circle that fits entirely inside the colony, equal to "
        "the maximum of the object's Euclidean distance transform. Attained at the "
        "colony's distance-transform peak, which is the center used for "
        "RobustMeanRadius and ReachRadius. For an ideal disk it equals the disk "
        "radius. Formerly reported under the name MaxRadius.",
        tier=1,
    )
    ROBUST_MEAN_RADIUS = Entry(
        "RobustMeanRadius",
        "Symmetrically trimmed mean of the colony's radial signature: the distance "
        "from the colony center to its boundary, resampled uniformly over a fixed "
        "number of directions. The center is the centroid of the distance-transform "
        "peak plateau, which always lies inside the colony. Because the signature is "
        "sampled by angle rather than along the boundary, a narrow protrusion "
        "contributes only its angular width, so a single runner or spur cannot "
        "dominate the estimate. For an ideal disk this equals the disk radius. "
        "Compare InscribedRadius (the smallest radius) and ReachRadius (the largest).",
        tier=1,
    )
    REACH_RADIUS = Entry(
        "ReachRadius",
        "Maximum of the colony's radial signature: the distance from the colony "
        "center to the furthest point on its boundary. A ReachRadius much larger "
        "than RobustMeanRadius indicates a protrusion, spur, or runner extending "
        "from an otherwise compact colony.",
        tier=1,
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
    BBOX_AREA = Entry(
        "BboxArea",
        "Area of the smallest rectangle that completely contains the colony. Represents the total spatial shape of the colony including any empty space. In high-throughput assays, this helps assess colony positioning and potential interference with neighboring colonies.",
        tier=1,
    )
    MAJOR_AXIS_LENGTH = Entry(
        "MajorAxisLength",
        "Length of the longest axis of the ellipse that best fits the colony shape. Represents the maximum colony dimension. In arrayed microbial growth, this measurement helps identify colonies that have grown beyond their intended grid positions.",
        tier=1,
    )
    MINOR_AXIS_LENGTH = Entry(
        "MinorAxisLength",
        "Length of the shortest axis of the ellipse that best fits the colony shape. Represents the minimum colony dimension. Together with major axis length, this helps characterize colony aspect ratio and growth anisotropy.",
        tier=1,
    )
    COMPACTNESS = Entry(
        "Compactness",
        r"Calculated as :math:`\frac{\text{Perimeter}^2}{4\pi*\text{Area}}`. Inverse of circularity (ranges from 1 for perfect circles to higher values for irregular shapes). Measures colony shape complexity - compact, circular colonies have values near 1, while irregular or filamentous colonies have much higher values.",
    )
    ORIENTATION = Entry(
        "Orientation",
        "Angle (in radians) between the colony's major axis and the horizontal axis. Measures colony alignment and growth directionality. Random orientations are typical for most bacterial colonies, while consistent orientations may indicate environmental gradients or mechanical stresses during plating.",
    )
