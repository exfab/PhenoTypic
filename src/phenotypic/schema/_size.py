"""The labels and descriptions of the size measurements."""

from ._change_notes import SIZE_SHAPE_SPLIT_NOTE
from ._measurement_info import Entry
from ._tiers import DirectPhenotype


class SIZE(DirectPhenotype):
    """Measure the key size magnitudes of each detected colony.

    Extract colony area, integrated intensity, perimeter, convex-hull and
    bounding-box areas, best-fit-ellipse axis lengths, and a family of radii
    measured from one center inside the colony. These are the starting
    measurements for growth and fitness comparisons; form descriptors
    (circularity, solidity, eccentricity, Feret diameters) live in
    :class:`SHAPE`.
    """

    @classmethod
    def category(cls):
        return "Size"

    @classmethod
    def change_note(cls) -> str:
        return SIZE_SHAPE_SPLIT_NOTE

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
    )
    INTEGRATED_INTENSITY = Entry(
        "IntegratedIntensity",
        r"The sum of the object's grayscale pixels. Calculated as "
        r":math:`\sum{\text{pixel values}} \times \text{area}`.",
    )
    PERIMETER = Entry(
        "Perimeter",
        "Total length of the colony's outer boundary in pixels. Measures colony edge complexity and surface irregularity. Smooth, circular colonies have shorter perimeters relative to their area compared to irregular or filamentous colonies.",
    )
    CONVEX_AREA = Entry(
        "ConvexArea",
        'Area of the smallest convex polygon that completely contains the colony, computed from the convex hull of its pixel centers. Represents the colony\'s "filled-in" appearance if all indentations and holes were removed. Because the hull passes through pixel centers it is slightly smaller than the pixel count of a convex colony. Useful for detecting colony spreading patterns or invasive growth characteristics.',
    )
    BBOX_AREA = Entry(
        "BboxArea",
        "Area of the smallest rectangle that completely contains the colony. Represents the total spatial shape of the colony including any empty space. In high-throughput assays, this helps assess colony positioning and potential interference with neighboring colonies.",
    )
    MAJOR_AXIS_LENGTH = Entry(
        "MajorAxisLength",
        "Length of the longest axis of the ellipse that best fits the colony shape. Represents the maximum colony dimension. In arrayed microbial growth, this measurement helps identify colonies that have grown beyond their intended grid positions.",
    )
    MINOR_AXIS_LENGTH = Entry(
        "MinorAxisLength",
        "Length of the shortest axis of the ellipse that best fits the colony shape. Represents the minimum colony dimension. Together with major axis length, this helps characterize colony aspect ratio and growth anisotropy.",
    )
    INSCRIBED_RADIUS = Entry(
        "InscribedRadius",
        "Radius of the largest circle that fits entirely inside the colony, equal to "
        "the maximum of the colony's Euclidean distance transform: the distance from "
        "the colony center (see MedianRadius) to its nearest edge. For an ideal disk "
        "it equals the disk radius. It reflects the colony's narrowest dimension, not "
        "its overall extent: an elongated colony reports half its width whatever its "
        "length (a 100 x 20 pixel colony reports 10), and a runner or spur leaves it "
        "unchanged. Use MaxRadius for overall extent. The image border counts as an "
        "edge.",
    )
    MEDIAN_RADIUS = Entry(
        "MedianRadius",
        "Median distance from the colony center to its boundary over the radial "
        "signature: the boundary distance sampled in equal angular directions "
        "(360 by default), keeping the outermost boundary crossing in each. The "
        "center is the centroid of the distance-transform peak plateau; it lies "
        "inside compact colonies, but for a ring-shaped colony it can fall in the "
        "central hole. For an ideal disk it equals the disk "
        "radius; for an ideal 100 x 20 pixel rectangle it is 14.1 pixels. This is "
        "not the value the retired Shape_MedianRadius carried (a median distance to "
        "the nearest edge, now Shape_MedianBoundaryDist).",
    )
    MEAN_RADIUS = Entry(
        "MeanRadius",
        "Mean distance from the colony center to its boundary over the radial "
        "signature (see MedianRadius for the center and sampling). For an ideal "
        "disk it equals the disk radius; for an ideal 100 x 20 pixel rectangle it "
        "is 21.0 pixels. A runner or spur pulls it upward in proportion to its "
        "angular width; use RobustMeanRadius for the compact body. This is not the "
        "value the retired Shape_MeanRadius carried (a mean distance to the nearest "
        "edge, now Shape_MeanBoundaryDist).",
    )
    ROBUST_MEAN_RADIUS = Entry(
        "RobustMeanRadius",
        "Symmetrically trimmed mean (20% from each end by default) of the radial "
        "signature (see MedianRadius). Because the signature is sampled by angle, a "
        "narrow runner contributes only its angular width and is trimmed away, so "
        "this estimates the typical radius of the colony's compact body. The trim "
        "treats genuine elongation the same way: an ideal 100 x 20 pixel rectangle "
        "gives 16.2 pixels against a MeanRadius of 21.0 pixels. For an ideal disk "
        "it equals the disk radius.",
    )
    MAX_RADIUS = Entry(
        "MaxRadius",
        "Largest distance from the colony center to its boundary over the radial "
        "signature (see MedianRadius): the colony's furthest reach. For an ideal "
        "disk it equals the disk radius; for an ideal 100 x 20 pixel rectangle it "
        "is 50.9 pixels. A MaxRadius far above RobustMeanRadius indicates a "
        "protrusion, spur, or runner. This is not the value the retired "
        "Shape_MaxRadius carried (the inscribed radius, now Size_InscribedRadius).",
    )


SIZE.__doc__ = f"{SIZE.__doc__}\n\n{SIZE.change_note()}"
