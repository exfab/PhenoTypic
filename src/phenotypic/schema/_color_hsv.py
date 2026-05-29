"""Per-object summary statistics in the HSV color space."""

from ._measurement_info import MeasurementInfo


class ColorHSV(MeasurementInfo):
    """Measure colony color statistics across multiple perceptual color spaces.

    Extract per-colony color features from CIE XYZ, chromaticity (xy),
    CIE Lab (perceptually uniform), and HSV color spaces. For each
    channel the standard statistical suite is computed (min, Q1, mean,
    median, Q3, max, std dev, coefficient of variation), plus Lab chroma
    estimates.

    Covers the HSV channels (Hue, Saturation, Brightness).
    """

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
