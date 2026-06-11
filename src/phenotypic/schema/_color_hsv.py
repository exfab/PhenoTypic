"""Per-object robust summary in the HSV color space (cone-embedded)."""

from ._measurement_info import MeasurementInfo


class ColorHSV(MeasurementInfo):
    """Robust HSV summary for a colony.

    HSV hue is circular and HSV is not perceptually uniform, so the robust
    center is computed as the geometric median of cone-Cartesian coordinates
    (S*V*cosθ, S*V*sinθ, V) and converted back to H,S,V. ``HSVConeVariance`` is
    the trace of the cone-Cartesian covariance.
    """

    @classmethod
    def category(cls):
        return "ColorHSV"

    HUE_ROBUST_MEAN = ("HueRobustMean", "Hue of the cone-embedded geometric-median robust center (circular-correct)")
    SATURATION_ROBUST_MEAN = ("SaturationRobustMean", "Saturation of the cone-embedded geometric-median robust center")
    VALUE_ROBUST_MEAN = ("ValueRobustMean", "Value (brightness) of the cone-embedded geometric-median robust center")
    HSV_CONE_VARIANCE = ("HSVConeVariance", "Trace of the HSV cone-Cartesian covariance (single 3D HSV spread scalar); spread about the arithmetic mean of the cone coordinates (NOT about the reported RobustMean center)")

    @classmethod
    def robust_headers(cls):
        return [
            str(cls.HUE_ROBUST_MEAN),
            str(cls.SATURATION_ROBUST_MEAN),
            str(cls.VALUE_ROBUST_MEAN),
            str(cls.HSV_CONE_VARIANCE),
        ]
