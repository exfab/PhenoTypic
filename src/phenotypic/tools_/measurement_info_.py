"""Centralized measurement information enumerations for the PhenoTypic library.

This module contains all MeasurementInfo subclasses used across the library,
standardizing measurement naming conventions, metadata, and documentation.
"""

from phenotypic.abc_._measurement_info import MeasurementInfo


class TEXTURE(MeasurementInfo):
    """Second-order texture features derived from the gray-level co-occurrence matrix (GLCM).

    All features assume normalized GLCMs computed at one or more pixel offsets and averaged
    across directions unless otherwise noted. Values depend on quantization, window size,
    and scale; interpret ranges comparatively within the same imaging setup.
    """

    @classmethod
    def category(cls) -> str:
        return "Texture"

    ANGULAR_SECOND_MOMENT = (
        "AngularSecondMoment",
        """Angular second moment (energy / uniformity). Measures the degree of local homogeneity
        (Σ p(i,j)²). High values → uniform texture (e.g., smooth, yeast-like colonies with consistent
        mycelial density). Low values → heterogeneous surfaces (e.g., sectored, wrinkled, or mixed
        sporulation zones). Reflects colony surface regularity rather than brightness.""",
    )

    CONTRAST = (
        "Contrast",
        """Contrast (local intensity variation; Σ (i–j)² p(i,j)). High values indicate strong gray-level
        differences (e.g., sharply defined rings, radial sectors, raised or folded regions). Low values
        indicate gradual tonal changes or uniformly pigmented colonies. Quantifies visual roughness
        and zonation amplitude.""",
    )

    CORRELATION = (
        "Correlation",
        """Linear gray-level correlation between neighboring pixels. Positive, high values suggest
        structured spatial dependence (e.g., oriented radial hyphae or concentric patterns); near-zero
        values indicate uncorrelated, disordered growth (e.g., diffuse cottony mycelium). Sensitive to
        illumination gradients and directional GLCM computation.""",
    )

    VARIANCE = (
        "HaralickVariance",
        """GLCM variance (Σ (i–μ)² p(i,j)). Captures spread of co-occurring gray-level pairs, distinct
        from raw intensity variance. High values → complex, multi-zone textures with variable
        hyphal/spore densities. Low values → consistent gray-level relationships and simpler colony
        surfaces.""",
    )

    INVERSE_DIFFERENCE_MOMENT = (
        "InverseDifferenceMoment",
        """Homogeneity (Σ p(i,j) / (1 + (i–j)²)). High values → smooth, locally uniform textures
        (e.g., glabrous colonies, uniform aerial mycelium). Low values → abrupt gray-level changes
        (e.g., granular sporulation, wrinkled surfaces). Typically inversely correlated with Contrast.""",
    )

    SUM_AVERAGE = (
        "SumAverage",
        """Mean of gray-level sums (Σ k·p_{x+y}(k)). Reflects the average intensity combination of
        neighboring pixels. In fungal colonies, can loosely parallel mean colony brightness when
        illumination and exposure are controlled, but remains a second-order rather than first-order
        intensity metric.""",
    )

    SUM_VARIANCE = (
        "SumVariance",
        """Variance of gray-level sum distribution. High values → heterogeneous brightness zones
        (e.g., alternating dense/sparse or pigmented/non-pigmented regions). Low values → uniform
        tone across the colony. Often correlated with Contrast; use comparatively within one setup.""",
    )

    SUM_ENTROPY = (
        "SumEntropy",
        """Entropy of the gray-level sum distribution. High values → diverse brightness combinations
        and irregular zonation. Low values → repetitive or periodic brightness patterns (e.g., evenly
        spaced rings). Indicates spatial unpredictability of summed intensities.""",
    )

    ENTROPY = (
        "Entropy",
        """Global GLCM entropy (–Σ p(i,j)·log p(i,j)). Measures total texture disorder and information
        content. High values → complex, irregular colony surfaces (powdery, fuzzy, or sectored growth).
        Low values → simple, smooth, predictable patterns (glabrous or uniform colonies). Sensitive to
        gray-level quantization and image dynamic range.""",
    )

    DIFFERENCE_VARIANCE = (
        "DiffVariance",
        """Variance of gray-level difference distribution. High values → mixture of smooth and textured
        regions (e.g., smooth margins with wrinkled centers). Low values → consistent edge content.
        Highlights heterogeneity in edge magnitude across the colony.""",
    )

    DIFFERENCE_ENTROPY = (
        "DiffEntropy",
        """Entropy of gray-level difference distribution. High values → irregular, unpredictable
        intensity transitions (e.g., random sporulation or uneven mycelial networks). Low values →
        regular periodic transitions (e.g., concentric zonation). Reflects randomness of local contrast
        rather than its magnitude.""",
    )

    IMC1 = (
        "InfoCorrelation1",
        """Information measure of correlation 1. Compares joint vs marginal entropies to quantify
        mutual dependence between gray levels. Positive values → structured, predictable textures
        (e.g., organized radial growth); near-zero → independence between adjacent regions.
        Direction of sign varies with implementation.""",
    )

    IMC2 = (
        "InfoCorrelation2",
        """Information measure of correlation 2 (√[1 – exp(–2 (H_xy2–H_xy))]). Always ≥ 0.
        Values approaching 1 → strong spatial dependence and organized architecture (e.g., symmetric
        rings, radial structure). Values near 0 → random, independent patterns. Captures nonlinear
        organization missed by linear correlation.""",
    )

    @classmethod
    def get_headers(cls, scale: int, matrix_name) -> list[str]:
        """Return full texture labels with angles in order 0, 45, 90, 135 for each feature and the
        average across degrees of each feature at the end."""
        angles = [0, 45, 90, 135]
        labels: list[str] = []
        for member in cls.get_labels():
            for angle in angles:
                labels.append(
                        f"{cls.category()}{matrix_name}_{member}-deg{angle:03d}-scale{scale:02d}"
                )

        for member in cls.get_labels():
            labels.append(
                    f"{cls.category()}{matrix_name}_{member}-avg-scale{scale:02d}"
            )
        return labels


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


class ColorComposition(MeasurementInfo):
    """Measurement info for perceptual color composition using 11-color model."""

    @classmethod
    def category(cls):
        return "ColorComposition"

    # Define the 11 color categories with descriptions
    BLACK_PCT = ("BlackPct", "Percentage of pixels classified as black (Value < 20)")
    WHITE_PCT = (
        "WhitePct",
        "Percentage of pixels classified as white (Saturation < 15, Value > 85)",
    )
    GRAY_PCT = (
        "GrayPct",
        "Percentage of pixels classified as gray (Saturation < 15, Value 20-85)",
    )
    PINK_PCT = (
        "PinkPct",
        "Percentage of pixels classified as pink (Red/Magenta hue, Saturation 20-60, Value > 80)",
    )
    BROWN_PCT = (
        "BrownPct",
        "Percentage of pixels classified as brown (Red/Orange hue, Value 20-60)",
    )
    RED_PCT = (
        "RedPct",
        "Percentage of pixels classified as red (Hue 0-15° or 345-360°)",
    )
    ORANGE_PCT = ("OrangePct", "Percentage of pixels classified as orange (Hue 15-45°)")
    YELLOW_PCT = ("YellowPct", "Percentage of pixels classified as yellow (Hue 45-75°)")
    GREEN_PCT = ("GreenPct", "Percentage of pixels classified as green (Hue 75-150°)")
    CYAN_PCT = ("CyanPct", "Percentage of pixels classified as cyan (Hue 150-180°)")
    BLUE_PCT = ("BluePct", "Percentage of pixels classified as blue (Hue 180-250°)")
    PURPLE_PCT = (
        "PurplePct",
        "Percentage of pixels classified as purple/magenta (Hue 250-345°)",
    )

    @classmethod
    def all_headers(cls):
        """Return all color composition measurement headers."""
        return [
            str(cls.BLACK_PCT),
            str(cls.WHITE_PCT),
            str(cls.GRAY_PCT),
            str(cls.PINK_PCT),
            str(cls.BROWN_PCT),
            str(cls.RED_PCT),
            str(cls.ORANGE_PCT),
            str(cls.YELLOW_PCT),
            str(cls.GREEN_PCT),
            str(cls.CYAN_PCT),
            str(cls.BLUE_PCT),
            str(cls.PURPLE_PCT),
        ]


class SIZE(MeasurementInfo):
    """The labels and descriptions of the size measurements."""

    @classmethod
    def category(cls):
        return "Size"

    AREA = (
        "Area",
        "Total number of pixels occupied by the microbial colony."
        "Larger areas typically indicate more robust growth or longer incubation times.",
    )
    INTEGRATED_INTENSITY = (
        "IntegratedIntensity",
        r"The sum of the object\'s grayscale pixels. Calculated as"
        r"$\sum{pixel values}*area$",
    )


class GRID_LINREG_STATS(MeasurementInfo):
    """Grid linear regression statistics and residual errors.

    Provides measurements for evaluating grid alignment quality and detecting off-grid
    colonies in arrayed microbial assays.
    """

    @classmethod
    def category(cls):
        return "GridLinReg"

    ROW_LINREG_M = (
        "RowM",
        "Slope of row-wise linear regression fit across column positions. Measures systematic drift in row alignment. Values near 0 indicate horizontal rows; non-zero values suggest rotational misalignment or systematic row curvature across the plate.",
    )
    ROW_LINREG_B = (
        "RowB",
        "Intercept of row-wise linear regression fit. Represents the expected row coordinate when column position is 0. Combined with slope, defines the expected row trend line for quality assessment and position prediction.",
    )
    COL_LINREG_M = (
        "ColM",
        "Slope of column-wise linear regression fit across row positions. Measures systematic drift in column alignment. Values near 0 indicate vertical columns; non-zero values suggest rotational misalignment or systematic column curvature across the plate.",
    )
    COL_LINREG_B = (
        "ColB",
        "Intercept of column-wise linear regression fit. Represents the expected column coordinate when row position is 0. Combined with slope, defines the expected column trend line for quality assessment and position prediction.",
    )
    PRED_RR = (
        "PredRR",
        "Predicted row coordinate from column-wise linear regression. Uses the column position and column regression parameters (ColM, ColB) to estimate where the row coordinate should be if the grid were perfectly aligned. Used for calculating residual errors and detecting misaligned colonies.",
    )
    PRED_CC = (
        "PredCC",
        "Predicted column coordinate from row-wise linear regression. Uses the row position and row regression parameters (RowM, RowB) to estimate where the column coordinate should be if the grid were perfectly aligned. Used for calculating residual errors and detecting misaligned colonies.",
    )
    RESIDUAL_ERR = (
        "ResidualError",
        "Euclidean distance between the actual colony centroid and the predicted position from linear regression. Quantifies how far each colony deviates from the expected grid pattern. High values indicate misdetections, off-grid growth, or local plate warping. Used by refinement operations to filter outliers and select the most plausible colony per grid cell.",
    )


class ColorXYZ(MeasurementInfo):
    @classmethod
    def category(cls):
        return "ColorXYZ"

    X_MINIMUM = ("CieXMin", "The minimum X value of the object in CIE XYZ color space")
    X_Q1 = (
        "CieXQ1",
        "The lower quartile (Q1) X value of the object in CIE XYZ color space",
    )
    X_MEAN = ("CieXMean", "The mean X value of the object in CIE XYZ color space")
    X_MEDIAN = ("CieXMedian", "The median X value of the object in CIE XYZ color space")
    X_Q3 = (
        "CieXQ3",
        "The upper quartile (Q3) X value of the object in CIE XYZ color space",
    )
    X_MAXIMUM = ("CieXMax", "The maximum X value of the object in CIE XYZ color space")
    X_STDDEV = (
        "CieXStdDev",
        "The standard deviation of the X value of the object in CIE XYZ color space",
    )
    X_COEFF_VARIANCE = (
        "CieXCoeffVar",
        "The coefficient of variation of the X value of the object in CIE XYZ color space",
    )

    @classmethod
    def cieX_headers(cls):
        return [
            str(cls.X_MINIMUM),
            str(cls.X_Q1),
            str(cls.X_MEAN),
            str(cls.X_MEDIAN),
            str(cls.X_Q3),
            str(cls.X_MAXIMUM),
            str(cls.X_STDDEV),
            str(cls.X_COEFF_VARIANCE),
        ]

    Y_MINIMUM = ("CieYMin", "The minimum Y value of the object in CIE XYZ color space")
    Y_Q1 = (
        "CieYQ1",
        "The lower quartile (Q1) Y value of the object in CIE XYZ color space",
    )
    Y_MEAN = ("CieYMean", "The mean Y value of the object in CIE XYZ color space")
    Y_MEDIAN = ("CieYMedian", "The median Y value of the object in CIE XYZ color space")
    Y_Q3 = (
        "CieYQ3",
        "The upper quartile (Q3) Y value of the object in CIE XYZ color space",
    )
    Y_MAXIMUM = ("CieYMax", "The maximum Y value of the object in CIE XYZ color space")
    Y_STDDEV = (
        "CieYStdDev",
        "The standard deviation of the Y value of the object in CIE XYZ color space",
    )
    Y_COEFF_VARIANCE = (
        "CieYCoeffVar",
        "The coefficient of variation of the Y value of the object in CIE XYZ color space",
    )

    @classmethod
    def cieY_headers(cls):
        return [
            str(cls.Y_MINIMUM),
            str(cls.Y_Q1),
            str(cls.Y_MEAN),
            str(cls.Y_MEDIAN),
            str(cls.Y_Q3),
            str(cls.Y_MAXIMUM),
            str(cls.Y_STDDEV),
            str(cls.Y_COEFF_VARIANCE),
        ]

    Z_MINIMUM = ("CieZMin", "The minimum Z value of the object in CIE XYZ color space")
    Z_Q1 = (
        "CieZQ1",
        "The lower quartile (Q1) Z value of the object in CIE XYZ color space",
    )
    Z_MEAN = ("CieZMean", "The mean Z value of the object in CIE XYZ color space")
    Z_MEDIAN = ("CieZMedian", "The median Z value of the object in CIE XYZ color space")
    Z_Q3 = (
        "CieZQ3",
        "The upper quartile (Q3) Z value of the object in CIE XYZ color space",
    )
    Z_MAXIMUM = ("CieZMax", "The maximum Z value of the object in CIE XYZ color space")
    Z_STDDEV = (
        "CieZStdDev",
        "The standard deviation of the Z value of the object in CIE XYZ color space",
    )
    Z_COEFF_VARIANCE = (
        "CieZCoeffVar",
        "The coefficient of variation of the Z value of the object in CIE XYZ color space",
    )

    @classmethod
    def cieZ_headers(cls):
        return [
            str(cls.Z_MINIMUM),
            str(cls.Z_Q1),
            str(cls.Z_MEAN),
            str(cls.Z_MEDIAN),
            str(cls.Z_Q3),
            str(cls.Z_MAXIMUM),
            str(cls.Z_STDDEV),
            str(cls.Z_COEFF_VARIANCE),
        ]


class Colorxy(MeasurementInfo):
    @classmethod
    def category(cls):
        return "Colorxy"

    x_MINIMUM = ("xMin", "The minimum chromaticity x coordinate of the object")
    x_Q1 = ("xQ1", "The lower quartile (Q1) chromaticity x coordinate of the object")
    x_MEAN = ("xMean", "The mean chromaticity x coordinate of the object")
    x_MEDIAN = ("xMedian", "The median chromaticity x coordinate of the object")
    x_Q3 = ("xQ3", "The upper quartile (Q3) chromaticity x coordinate of the object")
    x_MAXIMUM = ("xMax", "The maximum chromaticity x coordinate of the object")
    x_STDDEV = (
        "xStdDev",
        "The standard deviation of the chromaticity x coordinate of the object",
    )
    x_COEFF_VARIANCE = (
        "xCoeffVar",
        "The coefficient of variation of the chromaticity x coordinate of the object",
    )

    @classmethod
    def x_headers(cls):
        return [
            str(cls.x_MINIMUM),
            str(cls.x_Q1),
            str(cls.x_MEAN),
            str(cls.x_MEDIAN),
            str(cls.x_Q3),
            str(cls.x_MAXIMUM),
            str(cls.x_STDDEV),
            str(cls.x_COEFF_VARIANCE),
        ]

    y_MINIMUM = ("yMin", "The minimum chromaticity y coordinate of the object")
    y_Q1 = ("yQ1", "The lower quartile (Q1) chromaticity y coordinate of the object")
    y_MEAN = ("yMean", "The mean chromaticity y coordinate of the object")
    y_MEDIAN = ("yMedian", "The median chromaticity y coordinate of the object")
    y_Q3 = ("yQ3", "The upper quartile (Q3) chromaticity y coordinate of the object")
    y_MAXIMUM = ("yMax", "The maximum chromaticity y coordinate of the object")
    y_STDDEV = (
        "yStdDev",
        "The standard deviation of the chromaticity y coordinate of the object",
    )
    y_COEFF_VARIANCE = (
        "yCoeffVar",
        "The coefficient of variation of the chromaticity y coordinate of the object",
    )

    @classmethod
    def y_headers(cls):
        return [
            str(cls.y_MINIMUM),
            str(cls.y_Q1),
            str(cls.y_MEAN),
            str(cls.y_MEDIAN),
            str(cls.y_Q3),
            str(cls.y_MAXIMUM),
            str(cls.y_STDDEV),
            str(cls.y_COEFF_VARIANCE),
        ]


class ColorLab(MeasurementInfo):
    @classmethod
    def category(cls):
        return "ColorLab"

    L_STAR_MINIMUM = ("L*Min", "The minimum L* value of the object")
    L_STAR_Q1 = ("L*Q1", "The lower quartile (Q1) L* value of the object")
    L_STAR_MEAN = ("L*Mean", "The mean L* value of the object")
    L_STAR_MEDIAN = ("L*Median", "The median L* value of the object")
    L_STAR_Q3 = ("L*Q3", "The upper quartile (Q3) L* value of the object")
    L_STAR_MAXIMUM = ("L*Max", "The maximum L* value of the object")
    L_STAR_STDDEV = ("L*StdDev", "The standard deviation of the L* value of the object")
    L_STAR_COEFF_VARIANCE = (
        "L*CoeffVar",
        "The coefficient of variation of the L* value of the object",
    )

    @classmethod
    def l_star_headers(cls):
        return [
            str(cls.L_STAR_MINIMUM),
            str(cls.L_STAR_Q1),
            str(cls.L_STAR_MEAN),
            str(cls.L_STAR_MEDIAN),
            str(cls.L_STAR_Q3),
            str(cls.L_STAR_MAXIMUM),
            str(cls.L_STAR_STDDEV),
            str(cls.L_STAR_COEFF_VARIANCE),
        ]

    A_STAR_MINIMUM = ("a*Min", "The minimum a* value of the object")
    A_STAR_Q1 = ("a*Q1", "The lower quartile (Q1) a* value of the object")
    A_STAR_MEAN = ("a*Mean", "The mean a* value of the object")
    A_STAR_MEDIAN = ("a*Median", "The median a* value of the object")
    A_STAR_Q3 = ("a*Q3", "The upper quartile (Q3) a* value of the object")
    A_STAR_MAXIMUM = ("a*Max", "The maximum a* value of the object")
    A_STAR_STDDEV = ("a*StdDev", "The standard deviation of the a* value of the object")
    A_STAR_COEFF_VARIANCE = (
        "a*CoeffVar",
        "The coefficient of variation of the a* value of the object",
    )

    @classmethod
    def a_star_headers(cls):
        return [
            str(cls.A_STAR_MINIMUM),
            str(cls.A_STAR_Q1),
            str(cls.A_STAR_MEAN),
            str(cls.A_STAR_MEDIAN),
            str(cls.A_STAR_Q3),
            str(cls.A_STAR_MAXIMUM),
            str(cls.A_STAR_STDDEV),
            str(cls.A_STAR_COEFF_VARIANCE),
        ]

    B_STAR_MINIMUM = ("b*Min", "The minimum b* value of the object")
    B_STAR_Q1 = ("b*Q1", "The lower quartile (Q1) b* value of the object")
    B_STAR_MEAN = ("b*Mean", "The mean b* value of the object")
    B_STAR_MEDIAN = ("b*Median", "The median b* value of the object")
    B_STAR_Q3 = ("b*Q3", "The upper quartile (Q3) b* value of the object")
    B_STAR_MAXIMUM = ("b*Max", "The maximum b* value of the object")
    B_STAR_STDDEV = ("b*StdDev", "The standard deviation of the b* value of the object")
    B_STAR_COEFF_VARIANCE = (
        "b*CoeffVar",
        "The coefficient of variation of the b* value of the object",
    )

    @classmethod
    def b_star_headers(cls):
        return [
            str(cls.B_STAR_MINIMUM),
            str(cls.B_STAR_Q1),
            str(cls.B_STAR_MEAN),
            str(cls.B_STAR_MEDIAN),
            str(cls.B_STAR_Q3),
            str(cls.B_STAR_MAXIMUM),
            str(cls.B_STAR_STDDEV),
            str(cls.B_STAR_COEFF_VARIANCE),
        ]

    CHROMA_EST_MEAN = (
        "ChromaEstimatedMean",
        r"The mean chroma estimation of the object calculated using :math:`\(sqrt(a^{*}_{mean}^2 + b^{*}_{mean})^2}`",
    )
    CHROMA_EST_MEDIAN = (
        "ChromaEstimatedMedian",
        r"The median chroma estimation of the object using :math:`\sqrt({a*_{median}^2 + b*_{median})^2}`",
    )


class ColorHSV(MeasurementInfo):
    @classmethod
    def category(cls):
        return "ColorHSV"

    HUE_MINIMUM = ("HueMin", "The minimum hue of the object")
    HUE_Q1 = ("HueQ1", "The lower quartile (Q1) hue of the object")
    HUE_MEAN = ("HueMean", "The mean hue of the object")
    HUE_MEDIAN = ("HueMedian", "The median hue of the object")
    HUE_Q3 = ("HueQ3", "The upper quartile (Q3) hue of the object")
    HUE_MAXIMUM = ("HueMax", "The maximum hue of the object")
    HUE_STDDEV = ("HueStdDev", "The standard deviation of the hue of the object")
    HUE_COEFF_VARIANCE = (
        "HueCoeffVar",
        "The coefficient of variation of the hue of the object",
    )

    @classmethod
    def hue_headers(cls):
        return [
            str(cls.HUE_MINIMUM),
            str(cls.HUE_Q1),
            str(cls.HUE_MEAN),
            str(cls.HUE_MEDIAN),
            str(cls.HUE_Q3),
            str(cls.HUE_MAXIMUM),
            str(cls.HUE_STDDEV),
            str(cls.HUE_COEFF_VARIANCE),
        ]

    SATURATION_MINIMUM = ("SaturationMin", "The minimum saturation of the object")
    SATURATION_Q1 = ("SaturationQ1", "The lower quartile (Q1) saturation of the object")
    SATURATION_MEAN = ("SaturationMean", "The mean saturation of the object")
    SATURATION_MEDIAN = ("SaturationMedian", "The median saturation of the object")
    SATURATION_Q3 = ("SaturationQ3", "The upper quartile (Q3) saturation of the object")
    SATURATION_MAXIMUM = ("SaturationMax", "The maximum saturation of the object")
    SATURATION_STDDEV = (
        "SaturationStdDev",
        "The standard deviation of the saturation of the object",
    )
    SATURATION_COEFF_VARIANCE = (
        "SaturationCoeffVar",
        "The coefficient of variation of the saturation of the object",
    )

    @classmethod
    def saturation_headers(cls):
        return [
            str(cls.SATURATION_MINIMUM),
            str(cls.SATURATION_Q1),
            str(cls.SATURATION_MEAN),
            str(cls.SATURATION_MEDIAN),
            str(cls.SATURATION_Q3),
            str(cls.SATURATION_MAXIMUM),
            str(cls.SATURATION_STDDEV),
            str(cls.SATURATION_COEFF_VARIANCE),
        ]

    BRIGHTNESS_MINIMUM = ("BrightnessMin", "The minimum brightness of the object")
    BRIGHTNESS_Q1 = ("BrightnessQ1", "The lower quartile (Q1) brightness of the object")
    BRIGHTNESS_MEAN = ("BrightnessMean", "The mean brightness of the object")
    BRIGHTNESS_MEDIAN = ("BrightnessMedian", "The median brightness of the object")
    BRIGHTNESS_Q3 = ("BrightnessQ3", "The upper quartile (Q3) brightness of the object")
    BRIGHTNESS_MAXIMUM = ("BrightnessMax", "The maximum brightness of the object")
    BRIGHTNESS_STDDEV = (
        "BrightnessStdDev",
        "The standard deviation of the brightness of the object",
    )
    BRIGHTNESS_COEFF_VARIANCE = (
        "BrightnessCoeffVar",
        "The coefficient of variation of the brightness of the object",
    )

    @classmethod
    def brightness_headers(cls):
        return [
            str(cls.BRIGHTNESS_MINIMUM),
            str(cls.BRIGHTNESS_Q1),
            str(cls.BRIGHTNESS_MEAN),
            str(cls.BRIGHTNESS_MEDIAN),
            str(cls.BRIGHTNESS_Q3),
            str(cls.BRIGHTNESS_MAXIMUM),
            str(cls.BRIGHTNESS_STDDEV),
            str(cls.BRIGHTNESS_COEFF_VARIANCE),
        ]


class SHAPE(MeasurementInfo):
    """The labels and descriptions of the shape measurements."""

    @classmethod
    def category(cls):
        return "Shape"

    AREA = (
        "Area",
        "Total number of pixels occupied by the microbial colony. Represents colony biomass and growth extent on agar plates. Larger areas typically indicate more robust growth or longer incubation times.",
    )
    PERIMETER = (
        "Perimeter",
        "Total length of the colony's outer boundary in pixels. Measures colony edge complexity and surface irregularity. Smooth, circular colonies have shorter perimeters relative to their area compared to irregular or filamentous colonies.",
    )
    CIRCULARITY = (
        "Circularity",
        r"Calculated as :math:`\frac{4\pi*\text{Area}}{\text{Perimeter}^2}`. Measures how closely a colony approximates a perfect circle (value = 1). Values < 1 indicate irregular colony morphology, which may result from genetic mutations, environmental stress, or mixed microbial populations on agar plates.",
    )
    CONVEX_AREA = (
        "ConvexArea",
        'Area of the smallest convex polygon that completely contains the colony. Represents the colony\'s "filled-in" appearance if all indentations and holes were removed. Useful for detecting colony spreading patterns or invasive growth characteristics.',
    )
    MEDIAN_RADIUS = (
        "MedianRadius",
        "Median distance from colony center to edge across all directions. Provides a robust measure of typical colony size that is less sensitive to outliers than mean width. Particularly useful for colonies with uneven growth or sectoring.",
    )
    MEAN_RADIUS = (
        "MeanRadius",
        "Average distance from colony center to edge across all directions. Represents overall colony expansion rate. In arrayed growth assays, this correlates with microbial fitness and growth kinetics under controlled conditions.",
    )
    MAX_RADIUS = (
        "MaxRadius",
        "Maximum distance from colony center to edge across all directions. Represents the furthest extent of colony growth from its center. In arrayed microbial assays, this measurement helps identify asymmetric growth patterns or colonies extending toward neighboring positions.",
    )
    MIN_FERET_DIAMETER = (
        "MinFeretDiameter",
        "Minimum caliper diameter - the shortest distance between two parallel tangent lines touching opposite sides of the colony. Represents the narrowest dimension of the colony regardless of orientation. Useful for detecting elongated or irregular colony morphologies and measuring colony width.",
    )
    MAX_FERET_DIAMETER = (
        "MaxFeretDiameter",
        "Maximum caliper diameter - the longest distance between two parallel tangent lines touching opposite sides of the colony. Represents the maximum dimension of the colony regardless of orientation. Often exceeds major axis length for irregular shapes and helps quantify maximum colony extent.",
    )
    ECCENTRICITY = (
        "Eccentricity",
        "Measure of colony elongation, ranging from 0 (perfect circle) to 1 (highly elongated). Values near 0 indicate compact, radially symmetric growth typical of healthy bacterial colonies, while higher values may suggest directional growth, motility, or environmental gradients on the agar surface.",
    )
    SOLIDITY = (
        "Solidity",
        "Ratio of actual colony area to its convex hull area (Area/ConvexArea). Values near 1 indicate compact, solid colonies with minimal indentations. Lower values (< 0.9) may indicate invasive growth, colony spreading, or the presence of clearing zones around colonies.",
    )
    EXTENT = (
        "Extent",
        "Ratio of colony area to its bounding box area (ObjectArea/BboxArea). Measures how efficiently the colony fills its allocated space. Compact colonies have higher extent values, while spread-out or irregular colonies have lower values.",
    )
    BBOX_AREA = (
        "BboxArea",
        "Area of the smallest rectangle that completely contains the colony. Represents the total spatial shape of the colony including any empty space. In high-throughput assays, this helps assess colony positioning and potential interference with neighboring colonies.",
    )
    MAJOR_AXIS_LENGTH = (
        "MajorAxisLength",
        "Length of the longest axis of the ellipse that best fits the colony shape. Represents the maximum colony dimension. In arrayed microbial growth, this measurement helps identify colonies that have grown beyond their intended grid positions.",
    )
    MINOR_AXIS_LENGTH = (
        "MinorAxisLength",
        "Length of the shortest axis of the ellipse that best fits the colony shape. Represents the minimum colony dimension. Together with major axis length, this helps characterize colony aspect ratio and growth anisotropy.",
    )
    COMPACTNESS = (
        "Compactness",
        r"Calculated as :math:`\frac{\text{Perimeter}^2}{4\pi*\text{Area}}`. Inverse of circularity (ranges from 1 for perfect circles to higher values for irregular shapes). Measures colony shape complexity - compact, circular colonies have values near 1, while irregular or filamentous colonies have much higher values.",
    )
    ORIENTATION = (
        "Orientation",
        "Angle (in radians) between the colony's major axis and the horizontal axis. Measures colony alignment and growth directionality. Random orientations are typical for most bacterial colonies, while consistent orientations may indicate environmental gradients or mechanical stresses during plating.",
    )


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


class SYMMETRIC_RADIUS(MeasurementInfo):
    """Mask-based radial expansion measurements for colonies on solid media.

    Summarises each colony's radial growth through four scalars derived
    directly from the binary object mask: the inoculum core radius (shared
    with :class:`RADIAL_EXPANSION`), the radius at which growth stops being
    angularly symmetric, and mean / maximum radial extent beyond the core.

    Unlike :class:`RADIAL_EXPANSION`, no skeletonization, branch tracing, or
    runner-outlier statistics are involved — all values are computed from
    per-annulus mask geometry and a circular-statistics angular resultant
    length. Intended for users who care about colony-level expansion and the
    symmetry of that expansion, not individual hyphae.
    """

    @classmethod
    def category(cls) -> str:
        return "SymmetricRadius"

    CORE_RADIUS = (
        "CoreRadius",
        "Radius of the dense inoculum core, determined by PELT changepoint "
        "detection on the radial mask-density profile centered on the "
        "inoculum. Growth measurements are reported relative to this boundary.",
    )
    SYMMETRIC_RADIUS = (
        "SymmetricRadius",
        "Radial distance from the inoculum centroid at which colony growth "
        "ceases to be angularly uniform. Computed as the first radius past "
        "the core where the smoothed per-annulus circular mean resultant "
        "length of mask-boundary pixels exceeds the symmetry threshold. "
        "Equals the colony outer envelope when growth remains symmetric "
        "throughout.",
    )
    MEAN_EXPANSION = (
        "MeanExpansion",
        "Mean distance of mask-boundary pixels from the inoculum centroid, "
        "measured from the core boundary outward. Captures the typical "
        "radial extent of growth past the inoculum, averaged over all "
        "angular directions.",
    )
    MAX_EXPANSION = (
        "MaxExpansion",
        "Maximum distance of any mask pixel from the inoculum centroid, "
        "measured from the core boundary outward. Captures the farthest "
        "extent of growth past the inoculum.",
    )
    CORE_END_RADIUS = (
        "CoreEndRadius",
        "Mean radius of the inoculum core boundary derived from the per-angle "
        "bright/background ratio walk. Each of 360 1° angular sectors finds the "
        "outer edge of the contiguous core run (bright fraction >= tau_core); the "
        "reported value is the mean across sectors. Compare with CoreRadius (the "
        "global PELT changepoint) — close agreement indicates a well-formed core.",
    )
    DENSE_END_RADIUS = (
        "DenseEndRadius",
        "Mean outer radius of the dense branching zone, where mask-bright pixels "
        "dominate (bright fraction >= tau_sparse). Per-angle radii are capped at "
        "the SymmetricRadius and angularly median-smoothed before averaging.",
    )
    SPARSE_END_RADIUS = (
        "SparseEndRadius",
        "Mean outer radius of the sparse branching zone (= colony envelope inside "
        "the symmetric growth front). Equals min(objmask outer envelope, "
        "SymmetricRadius) per angle, averaged across 360 sectors.",
    )
    CORE_AREA = (
        "CoreArea",
        "Pixel^2 area of the inoculum core zone, integrated across the 360-sector "
        "polar polygon defined by the per-angle core radii.",
    )
    DENSE_AREA = (
        "DenseArea",
        "Pixel^2 area of the dense branching zone, the annular region between the "
        "per-angle core boundary and dense-branching boundary.",
    )
    SPARSE_AREA = (
        "SparseArea",
        "Pixel^2 area of the sparse branching zone, the annular region between the "
        "per-angle dense boundary and the symmetric-envelope outer boundary.",
    )


class INTENSITY(MeasurementInfo):
    @classmethod
    def category(cls):
        return "Intensity"

    INTEGRATED_INTENSITY = ("IntegratedIntensity", "The sum of the object's pixels")
    DENSITY = ("Density", "The ratio of the object's intensity to the max possible "
                          "intensity of the object")
    CONVEX_DENSITY = ("ConvexDensity", "The ratio of the objects intensity to the max "
                                       "possible intensity of the object's convex hull")
    MINIMUM_INTENSITY = ("MinimumIntensity", "The minimum intensity of the object")
    MAXIMUM_INTENSITY = ("MaximumIntensity", "The maximum intensity of the object")
    MEAN_INTENSITY = ("MeanIntensity", "The mean intensity of the object")
    MEDIAN_INTENSITY = ("MedianIntensity", "The median intensity of the object")
    STANDARD_DEVIATION_INTENSITY = (
        "StandardDeviationIntensity",
        "The standard deviation of the object",
    )
    COEFFICIENT_VARIANCE_INTENSITY = (
        "CoefficientVarianceIntensity",
        "The coefficient of variation of the object",
    )
    Q1_INTENSITY = (
        "LowerQuartileIntensity",
        "The lower quartile intensity of the object",
    )
    Q3_INTENSITY = (
        "UpperQuartileIntensity",
        "The upper quartile intensity of the object",
    )
    IQR_INTENSITY = (
        "InterquartileRangeIntensity",
        "The interquartile range of the object",
    )


class BBOX(MeasurementInfo):
    @classmethod
    def category(cls) -> str:
        return "Bbox"

    CENTER_RR = "CenterRR", "The row coordinate of the center of the bounding box."
    MIN_RR = "MinRR", "The smallest row coordinate of the bounding box."
    MAX_RR = "MaxRR", "The largest row coordinate of the bounding box."
    CENTER_CC = "CenterCC", " The column coordinate of the center of the bounding box."
    MIN_CC = "MinCC", " The smallest column coordinate of the bounding box."
    MAX_CC = "MaxCC", " The largest column coordinate of the bounding box."
    INTENSITY_WEIGHTED_CENTER_RR = (
        "IntensityWeightedCenterRR",
        "The intensity-weighted center row coordinate of the object "
        "(skimage ``centroid_weighted``).",
    )
    INTENSITY_WEIGHTED_CENTER_CC = (
        "IntensityWeightedCenterCC",
        "The intensity-weighted center column coordinate of the object "
        "(skimage ``centroid_weighted``).",
    )
    DIST_WEIGHTED_CENTER_RR = (
        "DistWeightedCenterRR",
        "Row coordinate of the per-object Euclidean-distance-transform "
        "maximum (deepest interior point of the object mask). Robust to thin "
        "filamentous extensions that pull intensity-weighted centroids "
        "off-body.",
    )
    DIST_WEIGHTED_CENTER_CC = (
        "DistWeightedCenterCC",
        "Column coordinate of the per-object Euclidean-distance-transform "
        "maximum (deepest interior point of the object mask). Robust to thin "
        "filamentous extensions that pull intensity-weighted centroids "
        "off-body.",
    )


class GRID(MeasurementInfo):
    """Constants for grid structure in the PhenoTypic module."""

    @classmethod
    def category(cls) -> str:
        return "Grid"

    ROW_NUM = "RowNum", "The row idx of the object"
    ROW_INTERVAL_START = (
        "RowIntervalStart",
        "The start of the row interval of the object",
    )
    ROW_INTERVAL_END = "RowIntervalEnd", "The end of the row interval of the object"

    COL_NUM = "ColNum", "The column idx of the object"
    COL_INTERVAL_START = (
        "ColIntervalStart",
        "The start of the column interval of the object",
    )
    COL_INTERVAL_END = "ColIntervalEnd", "The end of the column interval of the object"

    ROW_MAJOR_IDX = (
        "RowMajorIdx",
        "The row-major index of the object. Row major is the standard in most"
        " programming and data science array libraries. Used for indexing into"
        " 2D arrays.",
    )

    COL_MAJOR_IDX = (
        "ColMajorIdx",
        "The col-major index of the object in an array. Lab automation logic uses"
        " column-major (column-wise) indexing for well plate operations because"
        " 96-well plates are physically arranged with 8 rows"
        " (labeled A-H) and 12 columns (numbered 1-12), and this layout maps directly"
        " to how multichannel pipettes operate."
    )


class MODEL_METRICS(MeasurementInfo):
    """Generic fit-quality metrics and diagnostics shared by all ModelFitter subclasses.

    These columns are produced by any model fitter that wraps
    :func:`scipy.optimize.least_squares`, independent of the specific
    mathematical model. Subclass-specific fitted parameters live in the
    subclass's own MeasurementInfo class (e.g., ``LOG_GROWTH_MODEL``).
    """

    @classmethod
    def category(cls) -> str:
        return "ModelMetrics"

    # fit-quality metrics
    MAE = "MAE", "The mean absolute error"
    MSE = "MSE", "The mean squared error"
    RMSE = "RMSE", "The root mean squared error"
    R2 = "R2", "The coefficient of determination"

    # fit diagnostics
    NUM_SAMPLES = "NumSamples", "The number of samples used for model fitting"
    LOSS = "OptimizerLoss", "The loss of model fitting"
    STATUS = "OptimizerStatus", "The output of the optimizer status"


class LOG_GROWTH_MODEL(MeasurementInfo):
    @classmethod
    def category(cls) -> str:
        return "LogGrowthModel"

    R_FIT = "r", "The intrinsic growth rate"
    K_FIT = "K", "The carrying capacity"
    N0_FIT = "N0", "The initial number of the colony size metric being fitted"
    LAM = (
        "lambda",
        "The regularization factor applied to the max specific growth rate "
        "and initial population size",
    )
    BETA = (
        "beta",
        (
            "The penalty factor applied to relative difference of "
            "the carrying capacity from the largest measurement"
        ),
    )
    GROWTH_RATE = "µmax", "The growth rate of the colony calculated as (K*r)/4"
    K_MAX = "Kmax", "The upper bound of the carrying capacity for model fitting"


class LINEAR_SOFTPLUS_MODEL(MeasurementInfo):
    @classmethod
    def category(cls) -> str:
        return "LinearSoftplusModel"

    v = ("v", "The post-lag phase growth rate.")
    s0 = ("s0", "The initial size")
    lam = ("lambda", "The duration of the lag phase")
    alpha = ("alpha", "lag phase transition sharpness")
    smax = ("smax", "The carrying capacity if fitted")
    beta = (
        "beta",
        "user-provided saturation transition sharpness. Not fitted since "
        "this is not always biologically meaningful and adds complexity "
        "to the fitting.",
    )


class EDGE_CORRECTION(MeasurementInfo):
    """Measurement info container for edge correction analysis results.

    Provides metadata for measurement values produced by the EdgeCorrector,
    organizing corrected colony measurements under the "EdgeCorrection" category.
    """

    @classmethod
    def category(cls) -> str:
        return "EdgeCorrection"

    CORRECTED_CAP = "Cap", "The carrying capacity for the target measurement"
    NEW_VAL = "NewVal", "The new value of the target measurement"


class GRID_SPATIAL(MeasurementInfo):
    """Measurement info for spatial information for grid pinned colonies"""

    @classmethod
    def category(cls) -> str:
        return "GridSpatial"

    LEFT_NEIGHBOR_OBJ_LABEL = "LeftNeighborObjLabel", ("The object label of the left"
                                                       " neighbor colony")
    LEFT_DISTANCE = "LeftDistance", ("The distance of the left neighbor colony"
                                     " calculated using euclidean distance between"
                                     " bounding boxes")

    RIGHT_NEIGHBOR_OBJ_LABEL = "RightNeighborObjLabel", ("The object label of"
                                                         " the right neighbor colony")
    RIGHT_DISTANCE = (
        "RightDistance",
        "The distance of the right neighbor colony calculated"
        " using euclidean distance between bounding boxes"
    )
    ABOVE_NEIGHBOR_OBJ_LABEL = "AboveNeighborObjLabel", ("The object label of"
                                                         " the above neighbor colony")
    ABOVE_DISTANCE = (
        "AboveDistance",
        "The distance of the above neighbor colony calculated using euclidean"
        " distance between bounding boxes"
    )
    UNDER_NEIGHBOR_OBJ_LABEL = ("UnderNeighborObjLabel",
                                "The object label of the under neighbor colony")
    UNDER_DISTANCE = (
        "UnderDistance",
        "The distance of the under neighbor colony calculated using euclidean distance between bounding boxes"
    )
