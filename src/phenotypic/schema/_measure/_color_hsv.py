"""Per-object robust summary in the HSV color space (cone-embedded)."""

from .._base._measurement_info import Entry
from .._base._units import UNITS
from .._base._tiers import DescriptiveTrait


class ColorHSV(DescriptiveTrait):
    """Robust HSV summary for a colony.

    HSV hue is circular and HSV is not perceptually uniform, so the robust
    center is computed as the geometric median of cone-Cartesian coordinates
    (S*V*cosθ, S*V*sinθ, V) and converted back to H,S,V. ``HSVConeVariance`` is
    the trace of the cone-Cartesian covariance.
    """

    @classmethod
    def metric_family(cls):
        return "ColorHSV"

    @classmethod
    def default_unit(cls) -> UNITS:
        return UNITS.UNITLESS

    HUE_ROBUST_MEAN = Entry(
        "HueRobustMean",
        "Hue of the colony's robust center color, as a fraction of a full "
        "turn around the color wheel. It is the hue of the cone-embedded "
        "geometric-median robust center (circular-correct): the geometric "
        "median of the pixels at (S·V·cos 2πH, S·V·sin 2πH, V), which handles "
        "the wrap-around of hue at red.",
        unit=UNITS.TURN,
    )
    SATURATION_ROBUST_MEAN = Entry(
        "SaturationRobustMean",
        "Saturation of the colony's robust center color, from 0 for gray to 1 "
        "for a fully saturated color. It is the saturation of the "
        "cone-embedded geometric-median robust center.",
    )
    VALUE_ROBUST_MEAN = Entry(
        "ValueRobustMean",
        "Brightness (HSV value, from 0 to 1) of the colony's robust center "
        "color. It is the value of the cone-embedded geometric-median robust "
        "center.",
    )
    HSV_CONE_VARIANCE = Entry(
        "HSVConeVariance",
        "How spread out the colony's pixel colors are in the HSV cone, as the "
        "sum of the variances of the three cone coordinates. It is the trace "
        "of the HSV cone-Cartesian covariance (single 3D HSV spread scalar); "
        "spread about the arithmetic mean of the cone coordinates (NOT about "
        "the reported RobustMean center).",
    )

    @classmethod
    def robust_headers(cls):
        return [
            str(cls.HUE_ROBUST_MEAN),
            str(cls.SATURATION_ROBUST_MEAN),
            str(cls.VALUE_ROBUST_MEAN),
            str(cls.HSV_CONE_VARIANCE),
        ]
